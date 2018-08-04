#ifndef CAFFE_QUADRUPLET_LOSS_LAYER_HPP_
#define CAFFE_QUADRUPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace {

 template <typename Dtype>
 class QuadrupletLossLayer : public LossLayer<Dtype> {
  public:
   explicit QuadrupletLossLayer(const LayerParameter& param)
       : LossLayer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);
 
   virtual inline int ExactNumBottomBlobs() const { return 4; }
   virtual inline const char* type() const { return "QuadrupletLoss"; }
   /**
    * Unlike most loss layers, in the TripletLossLayer we can backpropagate
    * to the first three inputs.
    */
   virtual inline bool AllowForceBackward(const int bottom_index) const {
     return bottom_index != 3;
   }
 
  protected:
   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);
   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);
 
   virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 };

} //namespace 

#endif  // CAFFE_LOSS_LAYER_HPP_
