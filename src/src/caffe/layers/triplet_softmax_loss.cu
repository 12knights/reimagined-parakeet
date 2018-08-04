#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TripletSoftmaxLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  
	int channels = prob_.channels();
/*	//check softmax layer, passed
	for(int i=0; i<prob_.count()/2;i++)
	{
		Dtype hook=std::exp(bottom[0]->data_at(i,0,0,0))/(std::exp(bottom[0]->data_at(i,0,0,0))+std::exp(bottom[0]->data_at(i,1,0,0)));
		CHECK_LE(std::abs(hook-prob_.data_at(i,0,0,0)),1e-6);
	}
*/

	int num= bottom[1]->num();
  Dtype margin1 = this->layer_param_.hard_quadruplet_loss_param().margin1();

	Dtype  margin=0;
	Dtype same_mean =0, diff_mean=0;
	int same_num=0, diff_num=0;
	for(int i=0; i<bottom[1]->num(); i++ )
	{
		for(int j=0; j<bottom[1]->num(); j++)
		{
			if(bottom[1]->data_at(j,0,0,0)==bottom[1]->data_at(i,0,0,0) && i!=j){same_mean +=prob_.data_at(i*num+j,0,0,0);  same_num++;  }
			if(bottom[1]->data_at(j,0,0,0)!=bottom[1]->data_at(i,0,0,0)){diff_mean +=prob_.data_at(i*num+j,0,0,0);  diff_num++; }
		}                             
	}
	same_mean = same_mean/static_cast<Dtype>(same_num); 
	diff_mean = diff_mean/static_cast<Dtype>(diff_num);
	margin = diff_mean-same_mean;
	LOG(INFO)<<"margin:"<<margin;
	Dtype* bp_w=bp_w_.mutable_gpu_data();
	caffe_gpu_set(bp_w_.count(), Dtype(0.0), bp_w);
//caculate loss
  Dtype loss(0.0);
	for(int i=2; i<3; i++ )
	{
		for(int k=0; k<15; k++)
			LOG(INFO)<<"prob:"<<prob_.data_at(i*num+k,0,0,0)<<",   "<<prob_.data_at(i*num+k,1,0,0);
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
						Dtype mdist1 = std::max(margin1 + prob_.data_at(i*num+j,0,0,0) - prob_.data_at(i*num+k,0,0,0), Dtype(0.0));
						if(mdist1<1e-5){
						}
						else
						{
							Dtype weight =6.9249*log(mdist1/margin1+1)+0.2;
							//Dtype weight = std::exp(1.3737*mdist1/margin1)-0.95;
							caffe_gpu_add_scalar(1,weight*Dtype(exp(prob_.data_at(i*num+j,0,0,0))), bp_w+(i*num+j));
							caffe_gpu_add_scalar(1,weight*Dtype(exp(1- prob_.data_at(i*num+k,0,0,0))), bp_w+(i*num+k));      
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
						Dtype mdist2 = std::max(margin2 + prob_.data_at(i*num+j,0,0,0) - prob_.data_at(k*num+l,0,0,0), Dtype(0.0));
						if(mdist2<1e-5){
						//	caffe_gpu_add_scalar(1, Dtype(0), bp_w + (i*num+maxIdx)*channels);
						//	caffe_gpu_add_scalar(1,Dtype(0), bp_w+(i*num+minIdx)*channels );
						}
						else
						{
							caffe_gpu_add_scalar(1,Dtype(1), bp_w+(i*num+j)*channels );
							caffe_gpu_add_scalar(1,Dtype(-1), bp_w+(k*num+l)*channels );
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
	
	//LOG(INFO)<<"prob_.diff_at(row*num+col,0,0,0):"<<prob_.diff_at(row*num+col,0,0,0);

  loss = loss /static_cast<Dtype>(term_num_); //static_cast<Dtype>(prob_.num()-bottom[1]->num());
	caffe_gpu_set(1, loss, top[0]->mutable_gpu_data());

}

template <typename Dtype>
__global__ void LabelDiffForward(const int n,const int num, const Dtype* A_in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    	out[index] = A_in[index/num] == A_in[index%num];
  }
}


template <typename Dtype>
__global__ void Mul(const int n, const Dtype* A_in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    	out[index]=A_in[index/2]*out[index];
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads,
          const Dtype* label, Dtype* bottom_diff, const int dim) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    bottom_diff[index * dim + label_value ] -= 1;
  }
}

template <typename Dtype>
void TripletSoftmaxLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
		const int num=bottom[1]->num();
    const int dim = prob_.count() / (num*num);
    const int nthreads = num*num;
	
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* bp_w = bp_w_.mutable_gpu_data();
		
	
    // NOLINT_NEXT_LINE(whitespace/operators)
		Dtype* top_label= top_label_.mutable_gpu_data();

	  LabelDiffForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->num()), CAFFE_CUDA_NUM_THREADS>>>(
      		bottom[0]->num(),bottom[1]->num(), label, top_label);
    	CUDA_POST_KERNEL_CHECK;


    SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_label, bottom_diff, dim);
		

		Mul<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->num()*2), CAFFE_CUDA_NUM_THREADS>>>(
      		bottom[0]->num()*2, bp_w, bottom_diff);
    const Dtype loss_weight = top[0]->cpu_diff()[0] /static_cast<Dtype>(term_num_);
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
}
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletSoftmaxLossLayer);

}  // namespace caffe
