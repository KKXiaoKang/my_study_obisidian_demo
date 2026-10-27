"""Read-only structural check for the Obsidian paper graph (stdlib only)."""
from pathlib import Path
import collections
import json
import re

ROOT = Path(__file__).resolve().parents[1]
VAULT = ROOT.parent
manifest = json.loads((ROOT / '_数据/graph.json').read_text())
errors = []
files = [p for p in VAULT.rglob('*') if p.is_file() and not any(s in p.parts for s in ('.git', '.obsidian'))]
paths = {str(p.relative_to(VAULT)) for p in files}
names = collections.defaultdict(set)
for p in files:
    rel = str(p.relative_to(VAULT))
    names[p.name].add(rel)
    names[p.stem].add(rel)

def resolve(raw):
    target = raw.replace('\\|', '|').split('|')[0].split('#')[0].strip()
    if not target:
        return {'local-anchor'}
    for candidate in (target, target + '.md'):
        if candidate in paths:
            return {candidate}
    return names.get(target, set())

link_count = 0
for p in [VAULT / '00-论文知识图谱.md', *ROOT.rglob('*.md')]:
    text = re.sub(r'```[\s\S]*?```|~~~[\s\S]*?~~~', '', p.read_text())
    for match in re.finditer(r'!?\[\[([^\]\n]+)\]\]', text):
        link_count += 1
        if len(resolve(match[1])) != 1:
            errors.append(f'Unresolved/ambiguous link: {p.name}: {match[1]}')
    for line in text.splitlines():
        if line.startswith('|'):
            for match in re.finditer(r'\[\[[^\]]+\]\]', line):
                if re.search(r'(?<!\\)\|', match[0]):
                    errors.append(f'Unescaped table alias: {p.name}')

canvas_count = 0
for p in ROOT.rglob('*.canvas'):
    canvas_count += 1
    data = json.loads(p.read_text())
    nodes = data['nodes']
    ids = [n['id'] for n in nodes]
    if len(ids) != len(set(ids)):
        errors.append(f'Duplicate canvas node IDs: {p.name}')
    for n in nodes:
        if n['type'] == 'file' and n['file'] not in paths:
            errors.append(f'Missing canvas file: {n["file"]}')
        if not all(isinstance(n[k], (int, float)) for k in ('x', 'y', 'width', 'height')):
            errors.append(f'Invalid geometry: {p.name}/{n["id"]}')
        if n['type'] == 'file' and n.get('subpath'):
            heading = n['subpath'][1:]
            if not re.search(r'^#{1,6} ' + re.escape(heading) + r'\s*$', (VAULT/n['file']).read_text(), re.M):
                errors.append(f'Missing section: {p.name}/{n["id"]}')
    edge_ids = [e['id'] for e in data['edges']]
    if len(edge_ids) != len(set(edge_ids)):
        errors.append(f'Duplicate canvas edge IDs: {p.name}')
    for e in data['edges']:
        if e['fromNode'] not in ids or e['toNode'] not in ids:
            errors.append(f'Invalid edge: {p.name}/{e["id"]}')
    boxes = [n for n in nodes if n['type'] != 'group']
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            overlap = a['x'] < b['x']+b['width'] and b['x'] < a['x']+a['width'] and a['y'] < b['y']+b['height'] and b['y'] < a['y']+a['height']
            if overlap:
                errors.append(f'Overlapping cards: {p.name}/{a["id"]}/{b["id"]}')

node_ids = {n['id'] for n in manifest['nodes']}
covered = manifest['source_to_nodes']
if len(covered) != manifest['source_count']:
    errors.append('Source coverage count mismatch')
for p,keys in covered.items():
    if p not in paths or not keys or any(k not in node_ids for k in keys):
        errors.append(f'Invalid source coverage: {p}')
for e in manifest['edges']:
    if e['source'] not in node_ids or e['target'] not in node_ids or not e['reason']:
        errors.append('Invalid semantic relation')
for n in manifest['nodes']:
    if n['path'] not in paths:
        errors.append(f'Missing node file: {n["id"]}')
    if not any(n['id'] in [e['source'],e['target']] for e in manifest['edges']):
        errors.append(f'Isolated node: {n["id"]}')

result = dict(nodes=len(node_ids), relations=len(manifest['edges']), covered_sources=len(covered), markdown_links=link_count, canvases=canvas_count, errors=errors)
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(bool(errors))
