use crate::model::self_attention_multihead;
use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}
pub fn broadcast_shapes(shape1: &Vec<usize>, shape2: &Vec<usize>) -> Vec<usize> {
    let mut result = vec![];
    let len1 = shape1.len();
    let len2 = shape2.len();

    for i in 0..len1.max(len2) {
        let dim1 = if i >= len1 { 1 } else { shape1[len1 - 1 - i] };
        let dim2 = if i >= len2 { 1 } else { shape2[len2 - 1 - i] };

        if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
            panic!("Cannot broadcast shapes {:?} and {:?}", shape1, shape2);
        }
        result.push(dim1.max(dim2));
    }

    result.reverse();
    result
}

pub fn broadcast_tensor<T: Copy + Default>(
    tensor: &Tensor<T>,
    target_shape: &Vec<usize>,
) -> Tensor<T> {
    let source_shape = tensor.shape();
    let source_data = tensor.data();

    // 1) 先求目标形状的总元素数
    let broadcasted_size: usize = target_shape.iter().product();
    let mut broadcasted_data = vec![T::default(); broadcasted_size];

    // 2) 预计算源张量每个维度的 strides（从左到右，最后一个维度 stride=1）
    //    比如 source_shape = [2, 3, 4], strides = [3*4, 4, 1] = [12, 4, 1]
    let mut strides = vec![1; source_shape.len()];
    for i in (0..source_shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * source_shape[i + 1];
    }

    // 3) 额外检查：逐个维度（从右到左）判断是否可广播
    {
        let mut i_s = source_shape.len() as isize - 1;
        let mut i_t = target_shape.len() as isize - 1;
        while i_s >= 0 && i_t >= 0 {
            let sdim = source_shape[i_s as usize];
            let tdim = target_shape[i_t as usize];
            if sdim != 1 && tdim != 1 && sdim != tdim {
                panic!(
                    "无法将形状 {:?} 广播到 {:?}, 维度 {} 与 {} 不兼容",
                    source_shape, target_shape, sdim, tdim
                );
            }
            i_s -= 1;
            i_t -= 1;
        }
        // 注意：如果 source 维数比 target 维数更少，是允许的；那些更“高”位会被当作复制。
        // 若 target 更短，一般在外层逻辑就不会调用此函数去做“反向”广播，这里不再处理。
    }

    // 4) 遍历目标张量的每个索引 i（相当于 linear index），逐步解码到各维度，然后映射回源张量。
    //    解码顺序：先 / target_shape[last] 求出“在最后一维的坐标”，再 / target_shape[last-1]...
    for i in 0..broadcasted_size {
        let mut tmp = i;
        let mut source_index = 0;

        // 从右到左遍历 target_shape
        for dim_t in (0..target_shape.len()).rev() {
            let coord_t = tmp % target_shape[dim_t]; // 目标张量该维度的坐标
            tmp /= target_shape[dim_t];

            // 计算对应的源张量的维度下标 dim_s
            let offset = target_shape.len() - 1 - dim_t; // 与最右维的距离
            let dim_s = (source_shape.len() as isize - 1) - (offset as isize);
            if dim_s < 0 {
                // 源张量在该维度上相当于自动视为 1，所有坐标都映射到源张量的0
                continue;
            }

            let dim_s = dim_s as usize;
            let size_s = source_shape[dim_s];
            let stride_s = strides[dim_s];

            // 如果源张量在该维度 = 1，就只能用 index=0 做复制
            // 否则就用 coord_t
            let src_coord = if size_s == 1 { 0 } else { coord_t };

            source_index += src_coord * stride_s;
        }

        // 根据算出的 source_index 去源张量 data() 中取值
        broadcasted_data[i] = source_data[source_index];
    }

    // 5) 构造新的广播后 Tensor 返回
    Tensor::new(broadcasted_data, target_shape)
}

pub fn add_broadcasted<T: Copy + Default + std::ops::Add<Output = T>>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> Tensor<T> {
    let target_shape = broadcast_shapes(a.shape(), b.shape());
    let a_broadcasted = broadcast_tensor(a, &target_shape);
    let b_broadcasted = broadcast_tensor(b, &target_shape);

    let result_data: Vec<T> = a_broadcasted
        .data()
        .iter()
        .zip(b_broadcasted.data())
        .map(|(&x, &y)| x + y)
        .collect();

    Tensor::new(result_data, &target_shape)
}

// pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
//     // println!(
//     //     "Inside rms_norm: y.shape() = {:?}, x.shape() = {:?}",
//     //     y.shape(),
//     //     x.shape(),
//     // );
//     assert!(
//         y.shape() == x.shape(),
//         "Input and output tensors must have the same shape"
//     );
//     assert!(w.shape().len() == 1, "Weight tensor w must be 1D");
//
//     let shape = x.shape();
//     let last_dim = *shape.last().unwrap(); // 最后一个维度的大小
//     assert_eq!(
//         w.size(),
//         last_dim,
//         "Size of weight tensor must match the last dimension of input tensor"
//     );
//
//     let x_data = x.data();
//     let w_data = w.data();
//     let y_data = unsafe { y.data_mut() };
//
//     let batch_size = x.size() / last_dim;
//
//     for b in 0..batch_size {
//         let base = b * last_dim;
//
//         // 计算每个向量的均值平方根 (RMS)
//         let mean_square = (0..last_dim)
//             .map(|i| x_data[base + i].powi(2))
//             .sum::<f32>()
//             / last_dim as f32;
//         let rms = mean_square.sqrt().max(epsilon);
//
//         // 归一化并乘以权重
//         for i in 0..last_dim {
//             y_data[base + i] = (x_data[base + i] / rms) * w_data[i];
//         }
//     }
// }
pub fn rms_norm<T>(y: &mut Tensor<T>, x: &Tensor<T>, w: &Tensor<T>, epsilon: f32)
where
    T: Copy
        + Default
        + std::ops::Mul<Output = T>
        + std::ops::Div<Output = T>
        + Into<f32>
        + From<f32>,
{
    // 提前获取只读信息，避免后续对 y 的不可变借用
    let y_shape = y.shape().clone();
    let y_size = y.size();

    // 获取输入和输出张量的形状
    let x_shape = x.shape();
    let w_shape = w.shape();

    // 检查权重张量 w 必须是 1D
    assert_eq!(
        w_shape.len(),
        1,
        "Weight tensor w must be 1D, but got shape: {:?}",
        w_shape
    );

    // 检查权重张量的大小是否与最后一个维度匹配
    assert_eq!(
        w.size(),
        *y_shape.last().unwrap(),
        "Size of weight tensor must match the last dimension of output tensor"
    );

    // 如果输入形状与输出形状不一致，尝试广播输入张量
    let x_broadcasted = if x_shape != &y_shape {
        broadcast_tensor(x, &y_shape)
    } else {
        x.clone()
    };

    // 广播权重张量到输出张量的形状
    let w_broadcasted = broadcast_tensor(w, &vec![*y_shape.last().unwrap()]);

    // 获取张量数据
    let x_data = x_broadcasted.data();
    let w_data = w_broadcasted.data();

    // 提取 y 的可变数据
    let y_data = unsafe { y.data_mut() };

    let last_dim = *y_shape.last().unwrap();
    let batch_size = y_size / last_dim;

    for b in 0..batch_size {
        let base = b * last_dim;

        // 计算每个向量的均值平方根 (RMS)
        let mean_square: f32 = (0..last_dim)
            .map(|i| x_data[base + i].into().powi(2))
            .sum::<f32>()
            / last_dim as f32;
        let rms = mean_square.sqrt().max(epsilon);

        // 归一化并乘以权重
        for i in 0..last_dim {
            let x_val: f32 = x_data[base + i].into();
            let w_val: f32 = w_data[i].into();
            y_data[base + i] = T::from((x_val / rms) * w_val);
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    assert!(
        y.shape() == x.shape(),
        "Input and output tensors must have the same shape"
    );

    let length = y.size();
    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();

    for i in 0..length {
        let x_sigmoid = 1. / (1. + (-x_data[i]).exp());
        y_data[i] = y_data[i] * x_sigmoid * x_data[i];
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
// pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
//     let (m, k_a) = (a.shape()[0], a.shape()[1]); // A 的形状是 m x k_a
//     let (k_b, n) = (b.shape()[1], b.shape()[0]); // B 的形状是 n x k_b（转置后变为 k_b x n）
//     // 调试信息
//     // println!(
//     //     "Shape of A: {:?}, Shape of B: {:?}, Shape of C: {:?}",
//     //     a.shape(),
//     //     b.shape(),
//     //     c.shape()
//     // );
//     assert_eq!(k_a, k_b, "A 的列数应等于 B 转置的行数");
//     assert_eq!(c.shape()[0], m, "C 的行数应等于 A 的行数");
//     assert_eq!(c.shape()[1], n, "C 的列数应等于 B 的列数");
//
//     let a_data = a.data();
//     let b_data = b.data();
//     let c_data = unsafe { c.data_mut() };
//
//     for i in 0..m {
//         for j in 0..n {
//             let mut sum = 0.0;
//
//             // 计算 A 的第 i 行和 B 转置的第 j 行的点积
//             for k in 0..k_a {
//                 // B 转置等价于访问 b[j][k]
//                 sum += a_data[i * k_a + k] * b_data[j * k_b + k];
//             }
//
//             //  C[i][j]：beta * C[i][j] + alpha * sum
//             c_data[i * n + j] = beta * c_data[i * n + j] + alpha * sum;
//         }
//     }
//
// }

pub fn scale(tensor: &mut Tensor<f32>, factor: f32) {
    unsafe {
        for x in tensor.data_mut().iter_mut() {
            *x *= factor;
        }
    }
}

pub fn matmul_transb(
    c: &mut Tensor<f32>, // 输出矩阵 C
    beta: f32,           // 缩放因子，用于累加已有的输出
    a: &Tensor<f32>,     // 输入矩阵 A
    b: &Tensor<f32>,     // 输入矩阵 B，需要转置
    alpha: f32,          // 缩放因子，用于输出累积
) {
    // 获取形状信息
    let (m, k_a) = (a.shape()[0], a.shape()[1]);
    let (k_b, n) = (b.shape()[1], b.shape()[0]);
    let (m_c, n_c) = (c.shape()[0], c.shape()[1]);

    // 检查形状合法性
    assert_eq!(
        k_a, k_b,
        "A.cols must match B.cols for C = A x B^T! (A: [m,k], B: [n,k])"
    );
    assert_eq!(m, m_c, "C rows must match A rows!");
    assert_eq!(n, n_c, "C cols must match B rows (B is transposed)!");

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    // 计算 C = alpha * (A x B^T) + beta * C
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;

            // 计算 A 的第 i 行和 B 转置的第 j 行的点积
            for k in 0..k_a {
                let a_idx = i * k_a + k; // A 的索引
                let b_idx = j * k_b + k; // B^T 的索引 (B 的转置)

                // 检查索引是否越界
                assert!(
                    a_idx < a_data.len(),
                    "Index out of bounds for A: a_idx={}, len={}",
                    a_idx,
                    a_data.len()
                );
                assert!(
                    b_idx < b_data.len(),
                    "Index out of bounds for B: b_idx={}, len={}",
                    b_idx,
                    b_data.len()
                );

                sum += a_data[a_idx] * b_data[b_idx];
            }

            // 计算 C[i, j]
            let c_idx = i * n + j;
            assert!(
                c_idx < c_data.len(),
                "Index out of bounds for C: c_idx={}, len={}",
                c_idx,
                c_data.len()
            );
            c_data[c_idx] = alpha * sum + beta * c_data[c_idx];
        }
    }
}

/// 将张量中所有元素填充成同一个值
pub fn fill(t: &mut Tensor<f32>, value: f32) {
    let data = unsafe { t.data_mut() };
    for x in data.iter_mut() {
        *x = value;
    }
}

/// 将 src 的所有元素逐一复制到 dst，要求二者 size() 相同。
pub fn copy_slice(src: &Tensor<f32>, dst: &mut Tensor<f32>) {
    assert_eq!(
        src.size(),
        dst.size(),
        "copy_slice: src and dst must have the same number of elements"
    );

    let src_data = src.data();
    let dst_data = unsafe { dst.data_mut() };

    dst_data.copy_from_slice(src_data);
}

/// 普通矩阵乘法: C = alpha * (A @ B) + beta * C
/// A: shape = [M, K]
/// B: shape = [K, N]
/// C: shape = [M, N]
pub fn matmul(
    c: &mut Tensor<f32>, // 输出矩阵 C
    beta: f32,           // 原 C 的缩放系数
    a: &Tensor<f32>,     // 输入矩阵 A
    b: &Tensor<f32>,     // 输入矩阵 B
    alpha: f32,          // A @ B 的缩放系数
) {
    // 打印输入形状
    // println!(
    //     "Shape of A: {:?}, Shape of B: {:?}, Shape of C: {:?}",
    //     a.shape(),
    //     b.shape(),
    //     c.shape()
    // );

    // 断言形状合法性
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();

    assert_eq!(a_shape.len(), 2, "A must be a 2D matrix");
    assert_eq!(b_shape.len(), 2, "B must be a 2D matrix");
    assert_eq!(c_shape.len(), 2, "C must be a 2D matrix");

    let (m, k1) = (a_shape[0], a_shape[1]);
    let (k2, n) = (b_shape[0], b_shape[1]);
    let (cm, cn) = (c_shape[0], c_shape[1]);

    assert_eq!(k1, k2, "Inner dimensions of A and B must match");
    assert_eq!(m, cm, "Output rows must match A's rows");
    assert_eq!(n, cn, "Output columns must match B's columns");

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    // 矩阵乘法计算
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..k1 {
                sum += a_data[i * k1 + k] * b_data[k * n + j];
            }
            c_data[i * n + j] = alpha * sum + beta * c_data[i * n + j];
        }
    }
}

/// 元素相加：C = A + B
/// 要求 A, B, C 的形状完全相同
pub fn add(c: &mut Tensor<f32>, a: &Tensor<f32>, b: &Tensor<f32>) {
    // 1) 断言形状相同
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();

    assert_eq!(a_shape, b_shape, "A and B must have the same shape");
    assert_eq!(a_shape, c_shape, "A, B, and C must have the same shape");

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    // 2) 元素相加
    for i in 0..a_data.len() {
        c_data[i] = a_data[i] + b_data[i];
    }
}
pub fn subtract_max(data: &mut [f32], rows: usize, cols: usize) {
    for row in 0..rows {
        let row_start = row * cols;
        let row_end = row_start + cols;

        // 找到当前行的最大值
        let max_value = data[row_start..row_end]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        // 每个元素减去最大值
        for col in row_start..row_end {
            data[col] -= max_value;
        }
    }
}

pub fn add_in_place(a: &mut Tensor<f32>, b: &Tensor<f32>) {
    // 1) 断言形状相同
    let a_shape = a.shape();
    let b_shape = b.shape();

    assert_eq!(a_shape, b_shape, "A and B must have the same shape");

    let a_data = unsafe { a.data_mut() };
    let b_data = b.data();

    // 2) 元素相加
    for i in 0..a_data.len() {
        a_data[i] += b_data[i];
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
