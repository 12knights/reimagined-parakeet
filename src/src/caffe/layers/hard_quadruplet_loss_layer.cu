#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HardQuadrupletLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int channels = bottom[0]->channels();
	int num= bottom[1]->num();
  Dtype margin1 = this->layer_param_.hard_quadruplet_loss_param().margin1();
	Dtype margin2 = this->layer_param_.hard_quadruplet_loss_param().margin2();
	//	LOG(INFO)<<"margin1: "<<margin1<<", margin2: "<<margin2;
	caffe_gpu_set(bottom[0]->count(),Dtype(0.0), bottom[0]->mutable_gpu_diff());
//caculate loss
  	Dtype loss(0.0);

//test
/*	LOG(INFO)<<"bottom[0]->num()"<<bottom[0]->num();
	for(int i=0; i<bottom[1]->num(); i++ )
	{
		LOG(INFO)<<"label is: "<<bottom[1]->data_at(i,0,0,0);
	}
*/

	for(int i=2; i<3; i++ )
	{
		for(int k=0; k<15; k++)
			LOG(INFO)<<bottom[0]->data_at(i*num+k,0,0,0)<<",   "<<bottom[0]->data_at(i*num+k,1,0,0);
		LOG(INFO)<<"                                      ";
	}
//test end
	term_num_=0;
	for(int i=0; i<num; i++)
	{
		for(int  j=0; j<num; j++)
		{
			if(bottom[1]->data_at(j,0,0,0)==bottom[1]->data_at(i,0,0,0) && i!=j)  //same class
			{
				for(int k=0; k<num; k++)
				{
					if(bottom[1]->data_at(k,0,0,0)!=bottom[1]->data_at(i,0,0,0))
					{
						Dtype mdist1 = std::max(margin1 + bottom[0]->data_at(i*num+j,0,0,0) - bottom[0]->data_at(i*num+k,0,0,0), Dtype(0.0));
						if(mdist1<1e-5){
						//	caffe_gpu_add_scalar(1, Dtype(0), bottom[0]->mutable_gpu_diff() + (i*num+maxIdx)*channels);
						//	caffe_gpu_add_scalar(1,Dtype(0), bottom[0]->mutable_gpu_diff()+(i*num+minIdx)*channels );
						}
						else
						{
							Dtype weight =6.9249*log(mdist1/margin1+1)+0.2;//Dtype weight = std::exp(2.225*mdist1)-0.8;
							caffe_gpu_add_scalar(1,weight*Dtype(exp(prob_.data_at(i*num+j,0,0,0))), bottom[0]->mutable_gpu_diff()+(i*num+j)*channels );
							caffe_gpu_add_scalar(1,Dtype(-1)*weight*Dtype(exp(1- prob_.data_at(i*num+k,0,0,0))), bottom[0]->mutable_gpu_diff()+(i*num+k)*channels );
							term_num_++;
						}
						loss +=mdist1;	
						
					}// end for if k
				}// end for for k
			/*
			for(int k=0; k<num; k++)
			{
				for(int l=0; l<num; l++)
				{
					if(bottom[1]->data_at(k,0,0,0)!=bottom[1]->data_at(i,0,0,0) && bottom[1]->data_at(l,0,0,0)!=bottom[1]->data_at(i,0,0,0) && bottom[1]->data_at(k,0,0,0)!=bottom[1]->data_at(l,0,0,0))
					{
						Dtype mdist2 = std::max(margin2 + bottom[0]->data_at(i*num+j,0,0,0) - bottom[0]->data_at(k*num+l,0,0,0), Dtype(0.0));
						if(mdist2<1e-5){
						//	caffe_gpu_add_scalar(1, Dtype(0), bottom[0]->mutable_gpu_diff() + (i*num+maxIdx)*channels);
						//	caffe_gpu_add_scalar(1,Dtype(0), bottom[0]->mutable_gpu_diff()+(i*num+minIdx)*channels );
						}
						else
						{
							caffe_gpu_add_scalar(1,Dtype(1), bottom[0]->mutable_gpu_diff()+(i*num+j)*channels );
							caffe_gpu_add_scalar(1,Dtype(-1), bottom[0]->mutable_gpu_diff()+(k*num+l)*channels );
						}
						loss +=mdist2;
						term_num_++;
					}
				}// end for for l
			}//end for for k
			*/
			}	 //end for if j	
		}  //end for for j
  }  // end for for i
	
	//LOG(INFO)<<"bottom[0]->diff_at(row*num+col,0,0,0):"<<bottom[0]->diff_at(row*num+col,0,0,0);

  loss = loss /static_cast<Dtype>(term_num_); //static_cast<Dtype>(bottom[0]->num()-bottom[1]->num());
	caffe_gpu_set(1,loss, top[0]->mutable_gpu_data());

}

template <typename Dtype>
void HardQuadrupletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //Dtype margin = this->layer_param_.contrastive_loss_param().margin();
	int count = bottom[0]->count();
  Dtype alpha=top[0]->diff_at(0,0,0,0)/static_cast<Dtype>(term_num_);
	bottom[0]->scale_diff(alpha);	
}

INSTANTIATE_LAYER_GPU_FUNCS(HardQuadrupletLossLayer);

}  // namespace caffe
