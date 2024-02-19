# RelationalExperienceReplay-CL
This is an official PyTorch implementation of Relational Experience Replay: Continual Learning by Adaptively Tuning Task-wise Relationship.


Baseline DER++ on CIFAR-10 with buffer size M=200

> python utils/main.py --model derpp --dataset seq-cifar10 --lr 0.03 --batch_size 32 --minibatch_size 32 --n_epochs 50 --buffer_size 200 --seed <1000> --gpu_id <0> --exp <buf200>

Our method DER-CBA on CIFAR-10 with buffer size M=200

> python utils/main_relational.py --model r_der --dataset seq-cifar10 --lr 0.03 --batch_size 32 --minibatch_size 32 --n_epochs 50 --buffer_size 200 --seed <1000> --gpu_id <0> --exp <buf200>
