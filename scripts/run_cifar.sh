CUDA_VISIBLE_DEVICES=0,1 python hyperoptim.py -c "python main.py --dataset cifar --data /home/matteo/data/ --adv-train 0 --out-dir results/cifar-imb-new-cs/ --custom-schedule '[(-1,0.1),(80,0.01),(160,0.001),(180,1e-5)]' --epochs 200" -p '{"arch":["resnet18"],"cifar-imb":[0.005,0.01],"weight-decay":[0.0001],"inner-batch-factor":[1,2,3,4,5],"class-balanced":[0,1],"lr-cl":[1e-3,1e-4]}' -j 2 --exp_keys '["arch","cifar-imb","weight-decay","inner-batch-factor","class-balanced","lr-cl"]' -r
#CUDA_VISIBLE_DEVICES=0,1 python hyperoptim.py -c "python main.py --dataset cifar --data /home/matteo/data/ --adv-train 0 --out-dir results/cifar-imb-new/ --custom-schedule '[(-1,0.1),(80,0.01),(160,0.001),(180,1e-5)]' --epochs 200 --lr-cl 1e-5" -p '{"arch":["resnet34","resnet18"],"cifar-imb":[0.005,0.01,0.02,0.05,0.1,1.0],"weight-decay":[0.0001]}' -j 2 --exp_keys '["arch","cifar-imb","weight-decay"]' -r
#python hyperoptim.py -c "python -m exp_library.main --dataset cifar --data /home/matteo/data/ --adv-train 0 --out-dir results/cifar-imb/ --custom-schedule '[(-1,0.1),(80,0.01),(160,0.001),(180,1e-5)]' --epochs 200 --lr-cl 1e-5" -p '{"arch":["resnet18","resnet34"],"cifar-imb":[0.005,0.01,0.02,0.05,0.1,1.]}'