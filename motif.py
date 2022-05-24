import subprocess
class RunCmd(object):
  def cmd_run(self, cmd):
    self.cmd = cmd
    subprocess.call(self.cmd, shell=True)

for i in range(1):
    a = RunCmd()
    a.cmd_run('CUDA_VISIBLE_DEVICES=0 python crmsnet.py \
    --posi=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa \
    --nega=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa \
    --model_type=CRMSNet --motif=True  --n_epochs=40')

