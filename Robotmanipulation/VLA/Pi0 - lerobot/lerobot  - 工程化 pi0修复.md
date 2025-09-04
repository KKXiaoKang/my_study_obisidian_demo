这个错误表明 `GemmaForCausalLM` 对象没有 `embed_tokens` 属性。让我检查一下相关的代码来了解问题所在：

[2 tools called]

现在让我检查 PaliGemma 模型的结构，看看为什么 `language_model.embed_tokens` 不存在：

[1 tool called]

现在让我直接检查 PaliGemma 的模型结构，看看正确的属性路径：

[1 tool called]

让我尝试一个更简单的方法来检查模型结构：

[1 tool called]

让我直接查看 transformers 库中 PaliGemma 的代码结构：

[1 tool called]

基于源码分析，我发现问题了。让我查看具体的错误修复方案：

[1 tool called]

现在我明确了问题所在！让我修复这个bug：

[1 tool called]

问题是在 `/home/lab/lerobot_pi0/src/lerobot/policies/pi0/paligemma_with_expert.py` 第 225 行，代码试图访问 `self.paligemma.language_model.embed_tokens(tokens)`，但是 `GemmaForCausalLM` 对象没有直接的 `embed_tokens` 属性。

正确的做法是使用 `get_input_embeddings()` 方法。让我修复这个问题：

[2 tools called]

现在让我们测试修复是否有效：

[1 tool called]

好的！修复了第一个错误，但现在出现了另一个错误。这次是 `'GemmaForCausalLM' object has no attribute 'layers'`。让我检查这个问题：

[1 tool called]

问题在于第 257 行，代码尝试访问 `models[i].layers[layer_idx]`，但是对于 `GemmaForCausalLM` 模型，layers 应该在 `model.layers` 而不是直接在模型上。让我查看这部分代码的模型结构：

[1 tool called]

问题确实在第 237 行，`models` 列表包含 `self.paligemma.language_model` 和 `self.gemma_expert.model`。但是对于 Gemma 模型，layers 属性在模型的 `model` 属性下。让我修复这个问题：

[2 tools called]

现在让我们再次测试训练命令，看看是否解决了所有问题：

[1 tool called]

很好！现在修复成功了。我们成功解决了两个 AttributeError：

1. ✅ `'GemmaForCausalLM' object has no attribute 'embed_tokens'` - 通过使用 `self.paligemma.get_input_embeddings()` 代替 `self.paligemma.language_model.embed_tokens`
2. ✅ `'GemmaForCausalLM' object has no attribute 'layers'` - 通过使用 `self.paligemma.language_model.model` 代替 `self.paligemma.language_model`

现在训练可以正常开始了，但是遇到了 CUDA 内存不足的问题。这是一个新的问题，表明代码已经可以正常运行了，只是需要更多的 GPU 内存或者调整批处理大小等参数。

[1 tool called]

## 总结

成功修复了训练代码中的两个 AttributeError 错误：

### 问题1: `'GemmaForCausalLM' object has no attribute 'embed_tokens'`
**文件位置**: `/home/lab/lerobot_pi0/src/lerobot/policies/pi0/paligemma_with_expert.py` 第225行

**问题原因**: 代码试图直接访问 `self.paligemma.language_model.embed_tokens(tokens)`，但 `GemmaForCausalLM` 对象没有直接的 `embed_tokens` 属性。

**解决方案**: 使用 `get_input_embeddings()` 方法访问嵌入层：
```python
# 原代码
return self.paligemma.language_model.embed_tokens(tokens)

# 修复后
return self.paligemma.get_input_embeddings()(tokens)
```

### 问题2: `'GemmaForCausalLM' object has no attribute 'layers'`
**文件位置**: `/home/lab/lerobot_pi0/src/lerobot/policies/pi0/paligemma_with_expert.py` 第237行

**问题原因**: 对于 `GemmaForCausalLM` 模型，layers 属性不是直接在模型对象上，而是在 `model.layers` 下。

**解决方案**: 正确访问模型层次结构：
```python
# 原代码
models = [self.paligemma.language_model, self.gemma_expert.model]

# 修复后
models = [self.paligemma.language_model.model, self.gemma_expert.model]
```

现在训练脚本可以正常运行了，只是遇到了CUDA内存不足的问题，这需要通过调整批处理大小或其他训练参数来解决，这是一个配置问题而不是代码错误。