use std::{slice, sync::Arc, vec};

#[derive(Clone)]
pub struct Tensor<T> {
    data: Arc<Box<[T]>>,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

impl<T: Copy + Clone + Default> Tensor<T> {
    pub fn new(data: Vec<T>, shape: &Vec<usize>) -> Self {
        let length = data.len();
        Tensor {
            data: Arc::new(data.into_boxed_slice().try_into().unwrap()),
            shape: shape.clone(),
            offset: 0,
            length: length,
        }
    }

    pub fn default(shape: &Vec<usize>) -> Self {
        let length = shape.iter().product();
        let data = vec![T::default(); length];
        Self::new(data, shape)
    }

    pub fn data(&self) -> &[T] {
        &self.data[self.offset..][..self.length]
    }

    pub unsafe fn data_mut(&mut self) -> &mut [T] {
        let ptr = self.data.as_ptr().add(self.offset) as *mut T;
        slice::from_raw_parts_mut(ptr, self.length)
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn size(&self) -> usize {
        self.length
    }

    // Reinterpret the tensor as a new shape while preserving total size.
    pub fn reshape(&mut self, new_shape: &Vec<usize>) -> &mut Self {
        let new_length: usize = new_shape.iter().product();
        if new_length != self.length {
            let old_shape = self.shape.clone();
            panic!("New shape {new_shape:?} does not match tensor of {old_shape:?}");
        }
        self.shape = new_shape.clone();
        self
    }

    pub fn slice(&self, start: usize, shape: &Vec<usize>) -> Self {
        let new_length: usize = shape.iter().product();
        assert!(self.offset + start + new_length <= self.length);
        Tensor {
            data: self.data.clone(),
            shape: shape.clone(),
            offset: self.offset + start,
            length: new_length,
        }
    }

    pub fn add<F:Fn(T, T) -> T>(&self,  other: &Self, add_fn: F) -> Self {
        // 检查两个张量的形状是否一致
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensors must have the same shape to perform addition"
        );

        // 获取数据和大小
        let len = self.size();
        let self_data = self.data();
        let other_data = other.data();

        // 逐元素相加
        let result_data: Vec<T> = self_data
            .iter()
            .zip(other_data)
            .map(|(&x, &y)| add_fn(x, y))
            .collect();

        // 返回新的张量
        Tensor::new(result_data, self.shape())
    }

    /// 在 `dim` 维度上从 `start` 开始，长度为 `length`，返回一个切片后的新张量。
    /// 假设 `dim` < self.shape().len()，且范围合法。
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Self {
        // 当前形状
        let old_shape = self.shape();
        let rank = old_shape.len();
        assert!(dim < rank, "dim out of range");
        let end = start + length;
        assert!(end <= old_shape[dim], "narrow out of bounds");

        // 计算新shape
        let mut new_shape = old_shape.clone();
        new_shape[dim] = length;

        // 计算在 data() 上的偏移量:
        //   stride = shape[dim+1..].product() （如果是 row-major）
        //   offset_in_this_dim = start * stride
        //   plus self.offset
        // 例: shape=[n_heads, seq_len, dqkv], narrow(0, head, 1) => offset += head*(seq_len*dqkv)
        let mut stride: usize = 1;
        for s in old_shape[dim+1..].iter() {
            stride *= s;
        }
        let new_offset = self.offset + start * stride;

        // 创建新的 tensor
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            offset: new_offset,
            length: length * stride,
        }
    }

    /// squeeze 将 narrow 后某一维长度=1 的维度去除
    /// 如果你只想把 [1, seq_len, dqkv] => [seq_len, dqkv]
    /// 可以简单做:
    pub fn squeeze(&self, dim: usize) -> Self {
        assert!(dim < self.shape().len(), "dim out of range in squeeze");
        let mut new_shape = self.shape().clone();
        assert_eq!(new_shape[dim], 1, "squeeze a dimension whose size is not 1");
        new_shape.remove(dim);

        // length不变，但 offset也不变
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            offset: self.offset,
            length: self.length,
        }
    }

    pub fn transpose(&self) -> Self
    where
        T: Clone + Default,
    {
        let shape = self.shape();
        assert_eq!(shape.len(), 2, "transpose only supports 2D matrices");
        let (rows, cols) = (shape[0], shape[1]);

        let mut data = vec![T::default(); self.data().len()];
        for r in 0..rows {
            for c in 0..cols {
                data[c * rows + r] = self.data()[r * cols + c].clone();
            }
        }

        Self::new(data, &vec![cols, rows])
    }
    pub fn length(&self) -> usize {
        self.length
    }

}

// Some helper functions for testing and debugging
impl Tensor<f32> {
    #[allow(unused)]
    pub fn close_to(&self, other: &Self, rel: f32) -> bool {
        if self.shape() != other.shape() {
            return false;
        }
        let a = self.data();
        let b = other.data();
        
        return a.iter().zip(b).all(|(x, y)| float_eq(x, y, rel));
    }
    #[allow(unused)]
    pub fn print(&self){
        println!("shpae: {:?}, offset: {}, length: {}", self.shape, self.offset, self.length);
        let dim = self.shape()[self.shape().len() - 1];
        let batch = self.length / dim;
        for i in 0..batch {
            let start = i * dim;
            println!("{:?}", &self.data()[start..][..dim]);
        }
    }
}

#[inline]
pub fn float_eq(x: &f32, y: &f32, rel: f32) -> bool {
    (x - y).abs() <= rel * (x.abs() + y.abs()) / 2.0
}
