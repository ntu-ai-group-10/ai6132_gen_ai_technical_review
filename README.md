# AI6132 - Generative AI - Visual Synthesis Techniques Technical Review
This repository contains our work for the visual synthesis techniques technical review assignment from NTU AI6132 Generative AI class.

### Setup Instructions

1. Clone the repository

```
git clone https://github.com/ntu-ai-group-10/ai6132_gen_ai_technical_review.git
cd ai6132_gen_ai_technical_review
```

2. Create environment

```
conda create -n ai6132_gen_ai_technical_review python=3.9
conda activate ai6132_gen_ai_technical_review
conda install pytorch torchvision torchaudio torchmetrics pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

3. Prepare dataset

```
python prepare_dataset.py
```

4. Run experiments

```
python run.py --config config/stable_diffusion.yaml
```

5. Evaluate experiment results

```
python evaluation.py --config config/stable_diffusion.yaml
```
