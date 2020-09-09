# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""

import argparse
import os

import numpy as np
# # To get TF 1.x like behaviour in TF 2.0 one can run, instead of import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# import tensorflow as tf
import torch

from transformers import BertModel, BertForSequenceClassification, BertConfig


def convert_pytorch_checkpoint_to_tf(model: BertModel, ckpt_dir: str, model_name: str):

    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:

    Currently supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """

    tensors_to_transpose = ("dense.weight", "attention.self.query", "attention.self.key", "attention.self.value")

    var_map = (
        ("layer.", "layer_"),
        ("word_embeddings.weight", "word_embeddings"),
        ("position_embeddings.weight", "position_embeddings"),
        ("token_type_embeddings.weight", "token_type_embeddings"),
        (".", "/"),
        ("LayerNorm/weight", "LayerNorm/gamma"),
        ("LayerNorm/bias", "LayerNorm/beta"),
        ("weight", "kernel"),
        ("classifier/kernel", "output_weights"),#注意这里先被上面一行的("weight", "kernel")替换之后，在进行的("classifier/kernel", "output_weights")替换。原始变量名为classifier/weight
        ("classifier/bias", "output_bias"),
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()
    print("----------------------------")
    for var_name in state_dict:
        print(var_name)
    print("----------------------------")

    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        # return "bert/{}".format(name)
        return name

    #According to TF 1:1 Symbols Map, in TF 2.0 you should use tf.compat.v1.Session() instead of tf.Session()
    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        if os.path.isdir(model_name):
            model_name = os.path.basename(model_name)
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))

def main(raw_args=None):
    parser = argparse.ArgumentParser()
    base_name = r"E:\data\opensource_data\bert\bert_base_dir\output_jigsaw_sample_0.5"
    base_name = r"E:\code_cbg\bd_sparklesearch_mod\sparklesearch_modtext_bert\src\title_model_ori\tmp"
    model_name = "pytorch_model.bin"

    parser.add_argument("--model_name", default=base_name, type=str, required=False, help="model name e.g. bert-base-uncased")
    parser.add_argument(
        "--cache_dir", type=str, default=base_name, required=False, help="Directory containing pytorch model"
    )
    parser.add_argument("--pytorch_model_path", default=os.path.join(base_name, model_name), type=str, required=False, help="/path/to/<pytorch-model-name>.bin")
    parser.add_argument("--tf_cache_dir", default=os.path.join(base_name, "tmp"), type=str, required=False, help="Directory in which to save tensorflow model")
    args = parser.parse_args(raw_args)

    config = BertConfig.from_json_file(os.path.join(base_name, "bert_config.json"))
    model = BertForSequenceClassification(config)
    model_temp = torch.load(args.pytorch_model_path, map_location=torch.device('cpu'))
    model.load_state_dict(model_temp, strict=False)

    # model = BertModel.from_pretrained(
    #     pretrained_model_name_or_path=args.model_name,
    #     state_dict=torch.load(args.pytorch_model_path, map_location=torch.device('cpu')),
    #     cache_dir=args.cache_dir,
    # )

    convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir=args.tf_cache_dir, model_name=args.model_name)


if __name__ == "__main__":
    main()
