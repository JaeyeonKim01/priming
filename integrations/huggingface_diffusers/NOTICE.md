# Community attribution

The pipeline, transformer, and scheduler originate from
[BiliSakura/Priming-diffusers](https://huggingface.co/BiliSakura/Priming-diffusers/tree/a0c1743f49eb0f972feec280356ccaab90298b20),
contributed by [Bili-Sakura in issue #4](https://github.com/JaeyeonKim01/priming/issues/4).
The source revision and original file hashes are recorded in
[community-source.json](community-source.json).

The received source files carry Apache-2.0 headers, retained verbatim. Their
HuggingFace copyright notice is reproduced as received; it does not establish
upstream Diffusers acceptance. The repository [Apache-2.0 license](../../LICENSE)
applies to this adaptation.

Local changes: explicit scheduler component, standalone Hub packaging, preservation of caller autocast with FP32
weights, native-resolution restriction, rejection of unsupported guidance,
finite noise-scale validation, and callback fixes. The transformer and scheduler
architecture implementations are retained. Conversion, validation, tests, and
this documentation were added in this repository.
