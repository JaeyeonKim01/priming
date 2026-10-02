# JVP SDPA - Flash
This module contains an efficient implementation of the jacobian vector product (JVP) of scaled dot product attention.
This implementation is needed since pyTorch's built-in torch.nn.functional.scaled_dot_product_attention does not support jvp yet and vanilla attention consumes a lot of memory.

The module consists of 3 main triton kernels (+ a few helper kernels):
- fused normal + tangent (jvp) forward. Uses a modified version of [Ryu's triton kernel](https://github.com/Ryu1845/min-sCM/blob/main/standalone_multihead_jvp_test.py#L326).
- normal backward. Uses original [flash attention triton kernels from TriDao](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py)
- tangent backward. Custom built based on the math extracted from torch compile debug info and TriDao's backward kernel.

This is a very simple kernel that does not support masking, causal attention or custom softmax scales.
However, that functionality could easily be added.

In order to use it, simply invoke sdpa via
```
from sdpa_jvp.functional import sdpa_jvp
o = sdpa_jvp(q, k, v)
```


**IMPORTANT** Be aware that this kernel will silently produce incorrect results when used without JVP due to the forward of "normal" and "jvp" being fused into a single kernel.

## Benchmarks
JVP is on average roughly 3x as expensive as normal training.
Since JVP includes the "normal" forward pass, this means that the extra "tangent" graph is roughly 2x as expensive (1+2=3).

While the flash attention kernels help to get the peak memory utilization down signifcantly over the O(n²) vanilla implementation, it unfortunately fails to make it runtime significantly faster.
This is especially true for the backward pass of the jvp since it accumulates the gradients of many different paths in the graph -> requiring a lot of temporary memory which makes it difficult to fuse the operation into a efficient kernel.
In the best case, the results are accumulate in registers or SRAM ("shared memory"), but due to the large number of temporary variables used in the backward pass, this is not possible and parts of the intermediate results have to be cached in VRAM ("local memory") which is very slow in comparison.
While lowering the block sizes should in theory help to lower the memory consumption, it was not possible to find a size that does not spill registers - and very small block sizes are even slower.

### Forward pass
<table>
<tr><th>H</th><th>S</th><th>[ms] flash</th><th>[ms] vanilla</th><th>[MB] flash</th><th>[MB] vanilla</th><th>GFLOPs</th></tr>
<tr><td>1</td><td>128</td>      <td>1.07</td>  <td>1.12</td>  <td>32.25</td>  <td>32.50</td><td>0.025</td></tr>
<tr><td>1</td><td>1,024</td>    <td>1.06</td>  <td>1.12</td>  <td>34.01</td>  <td>46.51</td><td>1.610</td></tr>
<tr><td>1</td><td>4,096</td>    <td>1.04</td>  <td>1.19</td>  <td>40.05</td>  <td>234.0</td><td>25.77</td></tr>
<tr><td>1</td><td>8,192</td>    <td>1.48</td>  <td>1.98</td>  <td>48.09</td>  <td>820.1</td><td>103.1</td></tr>
<tr><td>1</td><td>16,384</td>   <td>2.58</td>  <td>6.76</td>  <td>64.19</td>  <td>3,144</td><td>412.3</td></tr>
<tr><td>1</td><td>32,768</td>   <td>11.34</td> <td>27.33</td> <td>96.38</td>  <td>12,400</td><td>1,649</td></tr>
<tr><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>24</td><td>128</td>      <td>1.02</td>  <td>1.09</td>  <td>38.04</td>  <td>44.02</td><td>0.604</td></tr>
<tr><td>24</td><td>1,024</td>    <td>1.01</td>  <td>1.13</td>  <td>80.28</td>  <td>380.1</td><td>38.66</td></tr>
<tr><td>24</td><td>4,096</td>    <td>4.25</td>  <td>10.59</td> <td>225.1</td>  <td>4,880</td><td>618.5</td></tr>
<tr><td>24</td><td>8,192</td>    <td>16.67</td> <td>45.25</td> <td>418.3</td>  <td>18,945</td><td>2,474</td></tr>
<tr><td>24</td><td>16,384</td>   <td>64.09</td> <td>161.1</td> <td>804.5</td>  <td>74,722</td><td>9,896</td></tr>
<tr><td>24</td><td>32,768</td>   <td>286.44</td><td>-</td>     <td>1,577</td>  <td>-</td>     <td>39,584</td></tr>
</table>

### Backward pass
Num heads, sequence length, duration, peak memory usage and theoretical FLOPs of vanilla implementation. All inputs use a batch size of 1 and a head dim of 128.
<table>
<tr><th>H</th><th>S</th><th>[ms] flash</th><th>[ms] vanilla</th><th>[MB] flash</th><th>[MB] vanilla</th><th>GFLOPs</th></tr>
<tr><td>1</td><td>128</td>  <td>1.09</td>   <td>1.51</td><td>64.69</td><td>64.80</td><td>0.050</td></tr>
<tr><td>1</td><td>1,024</td> <td>1.21</td>   <td>1.54</td><td>69.52</td><td>94.02</td><td>3.221</td></tr>
<tr><td>1</td><td>4,096</td> <td>2.06</td>   <td>1.53</td><td>86.06</td><td>508.1</td><td>51.5</td></tr>
<tr><td>1</td><td>8,192</td> <td>4.66</td>   <td>4.33</td><td>108.1</td><td>1,816</td><td>206</td></tr>
<tr><td>1</td><td>16,384</td><td>15.18</td>  <td>16.11</td><td>152.3</td><td>7,024</td><td>825</td></tr>
<tr><td>1</td><td>32,768</td><td>60.06</td>  <td>63.85</td><td>240.5</td><td>27,808</td><td>3,300</td></tr>
<tr><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>24</td><td>128</td>     <td>1.21</td>   <td>1.55</td>   <td>80.55</td>   <td>83.17</td><td>1.208</td></tr>
<tr><td>24</td><td>1,024</td>    <td>1.85</td>   <td>2.03</td>   <td>196.4</td>   <td>784.4</td><td>77.3</td></tr>
<tr><td>24</td><td>4,096</td>    <td>22.91</td>   <td>24.52</td>   <td>593.5</td>   <td>10,721</td><td>1,237</td></tr>
<tr><td>24</td><td>8,192</td>    <td>90.73</td>   <td>96.93</td>   <td>1,123</td>   <td>42,115</td><td>4,948</td></tr>
<tr><td>24</td><td>16,384</td>   <td>358.9</td>   <td>-</td>   <td>2,182</td>   <td>-</td><td>19,792</td></tr>
<tr><td>24</td><td>32,768</td>   <td>1,434</td>   <td>-</td>   <td>4,300</td>   <td>-</td><td>79,168</td></tr>
</table>