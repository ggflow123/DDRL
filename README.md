# DDRL

Repository for Data Distillation for Offline Reinforcement Learning. Feel free to check our[ website](https://datasetdistillation4rl.github.io) and [paper](https://openreview.net/forum?id=pnhiwoXZeY#).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate DDRL
```

### Hydra

The jobs are organized by Hydra, please refer to [Hydra](https://hydra.cc/docs/intro/) for more details about where the models are saved, especially during the 3rd step of **Data Distillation**.

# Data Distillation steps

### Environments:

We use three environments in Procgen benchmark: bigfish, starpilot and jumper, all in the easy mode with 200 seeds.

Environment names under configuration file:

```
env=bigfish200seeds # bigfish
env=jumper200seeds # jumper
env=starpilot200seeds # starpilot
```

### 1. Generate Offline Dataset:

Under the main repo, run

```bash
python src/offline_data.py model_save_path=bf200seeds.pt env=bigfish200seeds model_class=teacher.PPOTeacherProcgen
```

### 2. Generate Distilled Dataset:

Under the main repo, run:

```bash
python src/train_distill_data_and_save.py \
env=jumper200seeds \ #Change the environment along with teacher 
distill=exp10 \
distill.teacher_path=jp200seeds.pt \
distill.teacher_class=teacher.PPOTeacherProcgen \
loss=MSE student=policy_student_CNN \
trainablebuffer.buffer_kwargs.synthetic_buffer_size=15 \
trainablebuffer.buffer_kwargs.synthetic_init_threshold_size=1000 \
student_train_epochs=10 \
student_update_freq=1 \
distill.distill_kwargs.batch_size=75
```

### 3. Train Student Network with Distilled Dataset

Under the main repo, run:

```bash
python src/run_offline_datadistill.py \
env=bigfish200seeds \
distill=exp100 \
distill.teacher_path=bf200seeds.pt \
distill.teacher_class=teacher.PPOTeacherProcgen \
loss=MAE student=policy_student_CNN \
trainablebuffer.buffer_kwargs.synthetic_buffer_size=5000 \
trainablebuffer.buffer_kwargs.synthetic_init_threshold_size=1000 \
student_train_epochs=50 student_update_freq=1 \
distill.distill_kwargs.batch_size=256 # 75 for jumper and 2048 for starpilot
```

### 4. Evaluation

Put the trained student in the **models** folder, and run under the main repo:

```bash
python src/run_evaluation.py \
model_save_path=student-model__offline__starpilot200seeds__sp200seeds.pt__policy-student-CNN__CNN0__MAE__adam0__batch_buffer100__batch_trainablebuffer100__exp100.pt \
env=starpilot200seeds evaluator_class=evaluation.VecEvaluator
```
