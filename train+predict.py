import subprocess
class RunCmd(object):
  def cmd_run(self, cmd):
    self.cmd = cmd
    subprocess.call(self.cmd, shell=True)

for i in range(10):
    a = RunCmd()
    a.cmd_run('CUDA_VISIBLE_DEVICES=0 python crt.py \
    --posi=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa \
    --nega=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa \
    --model_type=CRTNet_4 --train=True --n_epochs=40')
    for j in range(10):
        a = RunCmd()
        a.cmd_run('CUDA_VISIBLE_DEVICES=0 python crt.py \
        --testfile=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa \
        --nega=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.negatives.fa \
        --model_type=CRTNet_4 --predict=True ')
