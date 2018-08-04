#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void QuadrupletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[1]->num(), bottom[2]->num());
	//CHECK_EQ(bottom[2]->num(), bottom[3]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
	CHECK_EQ(bottom[2]->channels(), 2);
 // CHECK_EQ(bottom[2]->channels(), bottom[3]->channels());
	CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[1]->height(), bottom[2]->height());
	CHECK_EQ(bottom[2]->height(), 1);
  //CHECK_EQ(bottom[2]->height(), bottom[3]->height());
	CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[1]->width(), bottom[2]->width());
	CHECK_EQ(bottom[1]->width(), 1);
  //CHECK_EQ(bottom[2]->width(), bottom[3]->width());

}

template <typename Dtype>
void QuadrupletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int channels = bottom[0]->channels();
  Dtype margin1 = this->layer_param_.quadruplet_loss_param().margin1();
	Dtype margin2 = this->layer_param_.quadruplet_loss_param().margin2();
	
//initiallize the input blob.diff
	//caffe_cpu_set(count, Dtype(0), bottom[0]->mutable_cpu_diff());
  //caffe_cpu_set(count, Dtype(0), bottom[1]->mutable_cpu_diff());

//caculate loss
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    const Dtype dist_ij=bottom[0]->data_at(i,0,0,0);
		const Dtype dist_ik=bottom[1]->data_at(i,0,0,0);
		const Dtype dist_kl=bottom[2]->data_at(i,0,0,0);
    Dtype mdist1 = std::max(margin1 + dist_ij - dist_ik, Dtype(0.0));
		Dtype mdist2 = std::max(margin2 + dist_ij - dist_kl, Dtype(0.0));
		if(mdist1==Dtype(0)){
    	//dist_binary_.mutable_cpu_data()[i] = Dtype(0);
    	//prepare for backward pass
			caffe_set(1, Dtype(0), bottom[0]->mutable_cpu_diff() + (i*channels));
  		caffe_set(1, Dtype(0), bottom[1]->mutable_cpu_diff() + (i*channels));
    }
		else
		{
			caffe_set(1, Dtype(1), bottom[0]->mutable_cpu_diff() + (i*channels));
  		caffe_set(1, Dtype(-1), bottom[1]->mutable_cpu_diff() + (i*channels));
		}
		if(mdist2==Dtype(0)){
    	//dist_binary_.mutable_cpu_data()[i] = Dtype(0);
    	//prepare for backward pass
	//		bottom[0]->mutable_cpu_diff()[i*channels]=Dtype(0)+ bottom[0]->mutable_cpu_diff()[i*channels];
		//	 bottom[2]->mutable_cpu_diff()[i*channels]= Dtype(0);
			caffe_set(1, Dtype(0), bottom[2]->mutable_cpu_diff()+i*channels);
    }
		else
		{
			caffe_add_scalar(1,Dtype(1),bottom[0]->mutable_cpu_diff()+i*channels);
			caffe_set(1, Dtype(-1),bottom[2]->mutable_cpu_diff()+i*channels);
			//bottom[0]->mutable_cpu_diff()[i*channels] = Dtype(1)+ bottom[0]->mutable_cpu_diff()[i*channels];
			//bottom[2]->mutable_cpu_diff()[i*channels] = Dtype(-1);
		}	

    loss =loss+mdist1+mdist2;
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num());
	caffe_set(1,loss, top[0]->mutable_cpu_data());	
}

template <typename Dtype>
void QuadrupletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 int count = bottom[0]->count();
  for (int i = 0; i < 3; ++i) {
      Dtype alpha=top[0]->diff_at(0,0,0,0)/static_cast<Dtype>(bottom[i]->num());
		caffe_cpu_axpby(count, alpha, bottom[i]->mutable_cpu_diff(), Dtype(0.0), bottom[i]->mutable_cpu_diff());
  } //for i
}

#ifdef CPU_ONLY
STUB_GPU(QuadrupletLossLayer);
#endif

INSTANTIATE_CLASS(QuadrupletLossLayer);
REGISTER_LAYER_CLASS(QuadrupletLoss);

}  // namespace caffe
