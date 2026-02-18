Extended Andrej Karpathy’s microgpt.py into a small experiment lab: reproducible runs, loss/metric tracking, train/val evaluation, and decoding controls (temperature/top-k/top-p) to understand how GPT training and generation behave. Karpathy is known for his work on deep learning education and for leading AI teams at Tesla and OpenAI.


What I learned so far.
- GPT training is next-token prediction: forward pass → cross-entropy loss → backprop → optimizer update.
- Training loss is noisy step-to-step; trends (moving averages/slope) are what matter.
- “Model quality” vs “output behavior” are separate: decoding settings can make the same model look repetitive, creative, or noisy.



