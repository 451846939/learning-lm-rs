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

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let shape = x.shape();
    let last_dim = *shape.last().unwrap(); // 最后一个维度的大小
    assert_eq!(
        w.size(),
        last_dim,
        "Size of weight tensor must match the last dimension of input tensor"
    );

    let x_data = x.data();
    let w_data = w.data();
    let y_data = unsafe { y.data_mut() };

    let batch_size = x.size() / last_dim;

    for b in 0..batch_size {
        let base = b * last_dim;

        // 计算每个向量的均值平方根 (RMS)
        let mean_square = (0..last_dim)
            .map(|i| x_data[base + i].powi(2))
            .sum::<f32>()
            / last_dim as f32;
        let rms = mean_square.sqrt().max(epsilon);

        // 归一化并乘以权重
        for i in 0..last_dim {
            y_data[base + i] = (x_data[base + i] / rms) * w_data[i];
        }
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    for i in 0..len {
        _y[i] *= _x[i] / (1. + (-_x[i]).exp());
    }

}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let (m, k_a) = (a.shape()[0], a.shape()[1]); // A 的形状是 m x k_a
    let (k_b, n) = (b.shape()[1], b.shape()[0]); // B 的形状是 n x k_b（转置后变为 k_b x n）
    // 调试信息
    // println!(
    //     "Shape of A: {:?}, Shape of B: {:?}, Shape of C: {:?}",
    //     a.shape(),
    //     b.shape(),
    //     c.shape()
    // );
    assert_eq!(k_a, k_b, "A 的列数应等于 B 转置的行数");
    assert_eq!(c.shape()[0], m, "C 的行数应等于 A 的行数");
    assert_eq!(c.shape()[1], n, "C 的列数应等于 B 的列数");

    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;

            // 计算 A 的第 i 行和 B 转置的第 j 行的点积
            for k in 0..k_a {
                // B 转置等价于访问 b[j][k]
                sum += a_data[i * k_a + k] * b_data[j * k_b + k];
            }

            //  C[i][j]：beta * C[i][j] + alpha * sum
            c_data[i * n + j] = beta * c_data[i * n + j] + alpha * sum;
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
    c: &mut Tensor<f32>,    // 输出矩阵 C
    beta: f32,              // 原 C 的缩放系数
    a: &Tensor<f32>,        // 输入矩阵 A
    b: &Tensor<f32>,        // 输入矩阵 B
    alpha: f32,             // A @ B 的缩放系数
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
