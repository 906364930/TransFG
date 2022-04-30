import torch
from models import configs
from models import modeling
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def my_save_model(my_model, path):
    path = path
    # absolute path end with .bin
    model_to_save = model.module if hasattr(my_model, 'module') else my_model
    checkpoint = {
        'model': model_to_save.state_dict(),
    }
    torch.save(checkpoint, path)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}

config = CONFIGS['ViT-B_16']
config.split = 'non-overlap'

test_input = torch.randn(5, 3, 448, 448)
test_hidden = torch.randn(5, 786, 768)
test_label = torch.tensor([0, 1, 2, 1, 0])

ori_block = modeling.Block(config)
pb = modeling.PrunedBlock(config)

model = modeling.VisionTransformer(config=config, img_size=448)
model_dict = torch.load("/home/luoqinglu/TransFg-with-git/weight/sample_run_checkpoint_combine_qkv.bin",
                        map_location=torch.device('cpu'))['model']
model.load_state_dict(model_dict)

test_out = model(test_input, test_label)
test_out2 = model(test_input)

# out_with_label = model(test_input, test_label)
# param = parameter_count_table(model)
# flops = FlopCountAnalysis(model, test_input)
# print("FLOPs: ", flops.total() / 5000000000)
print('end2')
