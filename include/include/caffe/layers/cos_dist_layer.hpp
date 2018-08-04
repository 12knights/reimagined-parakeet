#ifndef CAFFE_CROSS_ELTWISE_LAYER_HPP_
#define CAFFE_CROSS_ELTWISE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class CosDistLayer : public Layer<Dtype> {
 public:
  explicit CosDistLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CosDistLayer";}
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype>  bottom_norm_;
	Blob<Dtype>  bottom_square_;
	Blob<Dtype>  norm_;
	Blob<Dtype>  inner_prod_;
	Blob<Dtype>  ones_;
	Blob<Dtype>  ones_dim_;
	Blob<Dtype> diff1_;
	Blob<Dtype> diff2_;
	Blob<Dtype> identity_;
	Blob<Dtype> num_num_;
};

}  // namespace caffe
#endif  // CAFFE_SOFTMAX_LAYER_HPP_
