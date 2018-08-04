#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HardQuadrupletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	CHECK_EQ(bottom[0]->height(), 1);
	CHECK_EQ(bottom[0]->width(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	CHECK_EQ(bottom[0]->num(), bottom[1]->num()*bottom[1]->num());
	maxOrmin_.Reshape(bottom[1]->shape());
}

template <typename Dtype>
void HardQuadrupletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int channels = bottom[0]->channels();
	int num= bottom[1]->num();
  Dtype margin1 = this->layer_param_.hard_quadruplet_loss_param().margin1();
	Dtype margin2 = this->layer_param_.hard_quadruplet_loss_param().margin2();
	
//initiallize the input blob.diff
	//caffe_cpu_set(count, Dtype(0), bottom[0]->mutable_cpu_diff());
  //caffe_cpu_set(count, Dtype(0), bottom[1]->mutable_cpu_diff());
	Dtype* maxVec=maxOrmin_.mutable_cpu_data();
	caffe_set(bottom[0]->count(),Dtype(0), bottom[0]->mutable_cpu_diff());
//caculate loss
  Dtype loss(0.0);
	Dtype globalMin(1.0);
	std::vector<int > rows;
	std::vector<int > cols;
	int row=0, col=0;
	for(int i=0; i<num; i++)
	{
		int maxIdx=0, minIdx=0;
		Dtype maxNum = 0, minNum=1;
		for(int  j=0; j<num; j++)
		{
			if(bottom[1]->data_at(j,0,0,0)==bottom[1]->data_at(i,0,0,0) && i!=j && bottom[0]->data_at(i*num+j,0,0,0)>maxNum)  //same class
			{
				maxIdx= j;
				maxNum = bottom[0]->data_at(i*num+j,0,0,0);
			}
			else if(bottom[1]->data_at(j,0,0,0)!=bottom[1]->data_at(i,0,0,0) &&  bottom[0]->data_at(i*num+j,0,0,0)<minNum)   //different class
			{
				minIdx = j;
				minNum =  bottom[0]->data_at(i*num+j,0,0,0);
			}
		}
		rows.push_back(i);
		cols.push_back(maxIdx);
		Dtype mdist1 = std::max(margin1 + maxNum - minNum, Dtype(0.0));
		if(mdist1==Dtype(0)){
    	//dist_binary_.mutable_cpu_data()[i] = Dtype(0);
    	//prepare for backward pass
	//		caffe_add_scalar(1, Dtype(0), bottom[0]->mutable_cpu_diff() + (i*num+maxIdx)*channels);
	//		caffe_add_scalar(1,Dtype(0), bottom[0]->mutable_cpu_diff()+(i*num+minIdx)*channels );
    }
		else
		{
			caffe_add_scalar(1,Dtype(1), bottom[0]->mutable_cpu_diff()+(i*num+maxIdx)*channels );
			caffe_add_scalar(1,Dtype(-1), bottom[0]->mutable_cpu_diff()+(i*num+minIdx)*channels );
		}
		loss += mdist1;
		caffe_set(1, maxNum, maxVec+i);
		if(minNum<globalMin && bottom[1]->data_at(minIdx,0,0,0)==bottom[1]->data_at(i,0,0,0))
		{
			globalMin = minNum;
			row=i;
			col= minIdx;
		}
	}
	
  for (int i = 0; i < num; ++i) {
    const Dtype dist_ij=maxOrmin_.data_at(i,0,0,0);
		const Dtype dist_kl=globalMin;
		Dtype mdist2 = std::max(margin2 + dist_ij - dist_kl, Dtype(0.0));
		if(mdist2==Dtype(0)){
    }
		else
		{
			caffe_add_scalar(1,Dtype(1),bottom[0]->mutable_cpu_diff()+(rows[i]*num+cols[i])*channels);
			caffe_add_scalar(1, Dtype(-1),bottom[0]->mutable_cpu_diff()+(row*num+col)*channels);
		}	

    loss +=mdist2;
  }
  loss = loss / static_cast<Dtype>(bottom[1]->num());
	caffe_set(1,loss, top[0]->mutable_cpu_data());	
}



template <typename Dtype>
void HardQuadrupletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype alpha=top[0]->diff_at(0,0,0,0)/static_cast<Dtype>(bottom[0]->num());
	bottom[0]->scale_diff(alpha);		
}

#ifdef CPU_ONLY
STUB_GPU(HardQuadrupletLossLayer);
#endif

INSTANTIATE_CLASS(HardQuadrupletLossLayer);
REGISTER_LAYER_CLASS(HardQuadrupletLoss);

}  // namespace caffe
