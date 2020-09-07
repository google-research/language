#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("HasAnswer")
    .Input("blocks: string")
    .Input("answers: string")
    .Output("result : bool")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
HasAnswer.
)doc");

class HasAnswerOp : public tensorflow::OpKernel {
 public:
  explicit HasAnswerOp(tensorflow::OpKernelConstruction *context)
      : OpKernel(context) {}

  void Compute(tensorflow::OpKernelContext *context) override {
    TTypes<tstring>::ConstVec blocks = context->input(0).vec<tstring>();
    TTypes<tstring>::ConstVec answers = context->input(1).vec<tstring>();
    Tensor *outputs_t;
    OP_REQUIRES_OK(context, context->allocate_output(0, {blocks.dimension(0)},
                                                     &outputs_t));
    TTypes<bool>::Vec outputs = outputs_t->vec<bool>();
    for (int i = 0; i < blocks.dimension(0); ++i) {
      outputs(i) = FindAny(answers, blocks(i));
    }
  }

 private:
  bool FindAny(TTypes<tstring>::ConstVec &needles,
               const std::string &haystack) {
    for (int j = 0; j < needles.dimension(0); ++j) {
      if (haystack.find(needles(j)) != std::string::npos) {
        return true;
      }
    }
    return false;
  }
  TF_DISALLOW_COPY_AND_ASSIGN(HasAnswerOp);
};

REGISTER_KERNEL_BUILDER(Name("HasAnswer").Device(DEVICE_CPU), HasAnswerOp);

REGISTER_OP("ReaderInputs")
    .Input("question_token_ids: int32")
    .Input("block_token_ids: int32")
    .Input("block_lengths: int32")
    .Input("block_token_map: int32")
    .Input("answer_token_ids: int32")
    .Input("answer_lengths: int32")
    .Input("cls_token_id: int32")
    .Input("sep_token_id: int32")
    .Output("token_ids : int32")
    .Output("mask : int32")
    .Output("segment_ids : int32")
    .Output("block_mask : int32")
    .Output("token_map: int32")
    .Output("gold_starts : int32")
    .Output("gold_ends : int32")
    .Attr("max_sequence_len: int")
    .SetShapeFn([](shape_inference::InferenceContext *c) {
      const shape_inference::DimensionHandle num_blocks =
          c->Dim(c->input(2), 0);
      int max_sequence_len;
      TF_RETURN_IF_ERROR(c->GetAttr("max_sequence_len", &max_sequence_len));

      c->set_output(0, c->MakeShape({num_blocks, max_sequence_len}));
      c->set_output(1, c->MakeShape({num_blocks, max_sequence_len}));
      c->set_output(2, c->MakeShape({num_blocks, max_sequence_len}));
      c->set_output(3, c->MakeShape({num_blocks, max_sequence_len}));
      c->set_output(4, c->MakeShape({num_blocks, max_sequence_len}));
      c->set_output(
          5, c->MakeShape(
                 {num_blocks, shape_inference::InferenceContext::kUnknownDim}));
      c->set_output(
          6, c->MakeShape(
                 {num_blocks, shape_inference::InferenceContext::kUnknownDim}));
      return Status::OK();
    })
    .Doc(R"doc(
ReaderInputs.
)doc");

class ReaderInputsOp : public tensorflow::OpKernel {
 public:
  explicit ReaderInputsOp(tensorflow::OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("max_sequence_len", &max_sequence_len_));
  }

  void Compute(tensorflow::OpKernelContext *context) override {
    TTypes<int32>::ConstVec question_token_ids = context->input(0).vec<int32>();
    TTypes<int32>::ConstMatrix block_token_ids =
        context->input(1).matrix<int32>();
    TTypes<int32>::ConstVec block_lengths = context->input(2).vec<int32>();
    TTypes<int32>::ConstMatrix block_token_map =
        context->input(3).matrix<int32>();
    TTypes<int32>::ConstMatrix answer_token_ids =
        context->input(4).matrix<int32>();
    TTypes<int32>::ConstVec answer_lengths = context->input(5).vec<int32>();
    int32 cls_token_id = context->input(6).scalar<int32>()();
    int32 sep_token_id = context->input(7).scalar<int32>()();

    int num_blocks = block_token_ids.dimension(0);
    int question_length = question_token_ids.dimension(0);
    int num_answers = answer_token_ids.dimension(0);

    Tensor *token_ids_t;
    Tensor *mask_t;
    Tensor *segment_ids_t;
    Tensor *block_mask_t;
    Tensor *token_map_t;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, {num_blocks, max_sequence_len_},
                                            &token_ids_t));
    OP_REQUIRES_OK(context, context->allocate_output(
                                1, {num_blocks, max_sequence_len_}, &mask_t));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, {num_blocks, max_sequence_len_},
                                            &segment_ids_t));
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, {num_blocks, max_sequence_len_},
                                            &block_mask_t));
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, {num_blocks, max_sequence_len_},
                                            &token_map_t));
    TTypes<int32>::Matrix token_ids = token_ids_t->matrix<int32>();
    TTypes<int32>::Matrix mask = mask_t->matrix<int32>();
    TTypes<int32>::Matrix segment_ids = segment_ids_t->matrix<int32>();
    TTypes<int32>::Matrix block_mask = block_mask_t->matrix<int32>();
    TTypes<int32>::Matrix token_map = token_map_t->matrix<int32>();

    std::vector<std::vector<std::pair<int, int>>> spans;
    int max_answers = 0;
    for (int b = 0; b < num_blocks; ++b) {
      std::vector<int> token_ids_vec;
      token_ids_vec.reserve(max_sequence_len_);
      for (int i = 0; i < max_sequence_len_; ++i) {
        int question_index = i - 1;
        int block_index = i - 1 - question_length - 1;
        if (i == 0) {
          token_ids_vec.emplace_back(cls_token_id);
          token_ids(b, i) = cls_token_id;
          mask(b, i) = 1;
          segment_ids(b, i) = 0;
          block_mask(b, i) = 0;
          token_map(b, i) = -1;
        } else if (question_index < question_length) {
          token_ids_vec.emplace_back(question_token_ids(question_index));
          token_ids(b, i) = question_token_ids(question_index);
          mask(b, i) = 1;
          segment_ids(b, i) = 0;
          block_mask(b, i) = 0;
          token_map(b, i) = -1;
        } else if (question_index == question_length) {
          token_ids_vec.emplace_back(sep_token_id);
          token_ids(b, i) = sep_token_id;
          mask(b, i) = 1;
          segment_ids(b, i) = 0;
          block_mask(b, i) = 0;
          token_map(b, i) = -1;
        } else if (block_index <= block_lengths(b)) {
          if (block_index == block_lengths(b) || i == max_sequence_len_ - 1) {
            token_ids_vec.emplace_back(sep_token_id);
            token_ids(b, i) = sep_token_id;
            mask(b, i) = 1;
            segment_ids(b, i) = 1;
            block_mask(b, i) = 0;
            token_map(b, i) = -1;
          } else {
            token_ids_vec.emplace_back(block_token_ids(b, block_index));
            token_ids(b, i) = block_token_ids(b, block_index);
            mask(b, i) = 1;
            segment_ids(b, i) = 1;
            block_mask(b, i) = 1;
            token_map(b, i) = block_token_map(b, block_index);
          }
        } else {
          token_ids(b, i) = 0;
          mask(b, i) = 0;
          segment_ids(b, i) = 0;
          block_mask(b, i) = 0;
          token_map(b, i) = -1;
        }
      }

      spans.emplace_back();
      for (int j = 0; j < num_answers; ++j) {
        std::vector<int> answer_vec;
        answer_vec.reserve(answer_lengths(j));
        for (int k = 0; k < answer_lengths(j); ++k) {
          answer_vec.emplace_back(answer_token_ids(j, k));
        }
        const auto answer_search_end = token_ids_vec.end() - 1;
        auto answer_search_start =
            token_ids_vec.begin() + 1 + question_length + 1;
        while (answer_search_start != answer_search_end) {
          answer_search_start =
              std::search(answer_search_start, answer_search_end,
                          answer_vec.begin(), answer_vec.end());
          if (answer_search_start != answer_search_end) {
            int answer_start = answer_search_start - token_ids_vec.begin();
            int answer_end = answer_start + answer_vec.size() - 1;
            spans.back().emplace_back(answer_start, answer_end);
            ++answer_search_start;
            if (spans.back().size() > max_answers) {
              max_answers = spans.back().size();
            }
          }
        }
      }
    }

    Tensor *gold_starts_t;
    Tensor *gold_ends_t;
    OP_REQUIRES_OK(context, context->allocate_output(
                                5, {num_blocks, max_answers}, &gold_starts_t));
    OP_REQUIRES_OK(context, context->allocate_output(
                                6, {num_blocks, max_answers}, &gold_ends_t));
    TTypes<int>::Matrix gold_starts = gold_starts_t->matrix<int>();
    TTypes<int>::Matrix gold_ends = gold_ends_t->matrix<int>();
    for (int b = 0; b < num_blocks; ++b) {
      const std::vector<std::pair<int, int>> &current_spans = spans.at(b);
      for (int i = 0; i < max_answers; ++i) {
        if (i < current_spans.size()) {
          const std::pair<int, int> &current_span = current_spans.at(i);
          gold_starts(b, i) = current_span.first;
          gold_ends(b, i) = current_span.second;
        } else {
          gold_starts(b, i) = -1;
          gold_ends(b, i) = -1;
        }
      }
    }
  }

 private:
  int max_sequence_len_;
  TF_DISALLOW_COPY_AND_ASSIGN(ReaderInputsOp);
};

REGISTER_KERNEL_BUILDER(Name("ReaderInputs").Device(DEVICE_CPU),
                        ReaderInputsOp);

}  // namespace tensorflow
