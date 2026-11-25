import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from cd.train import train
from cd.evaluate import evaluate, predict
from cd.model import ChangeDetector
from cd.data import PairDataset
from cd.prepare import prepare
from cd.losses import pseudo_loss

class V2Tests(unittest.TestCase):
    def fixture(self,root):
        rng=np.random.default_rng(24)
        for split,n in [('train',20),('val',2),('test',2)]:
            for folder in ('A','B','label'):(root/split/folder).mkdir(parents=True)
            for i in range(n):
                for k in ('A','B'):Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(root/split/k/f'{i}.png')
                mask=np.zeros((32,32),np.uint8);mask[8:24,8:24]=255
                Image.fromarray(mask).save(root/split/'label'/f'{i}.png')

    def test_cached_crops_match_uncached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            records=[{'a':str(root/'train/A'/f'{i}.png'),'b':str(root/'train/B'/f'{i}.png'),'label':str(root/'train/label'/f'{i}.png')} for i in range(3)]
            left=PairDataset(records,16,seed=9,cache_pairs=0)
            right=PairDataset(records,16,seed=9,cache_pairs=2)
            for i in [0,0,1,0,2,1,0]:
                self.assertTrue(all(torch.equal(x,y) for x,y in zip(left[i],right[i])))

    def test_batch_inference_matches_single(self):
        torch.set_num_threads(2);torch.manual_seed(3)
        model=ChangeDetector().eval();rng=np.random.default_rng(3)
        a=Image.fromarray(rng.integers(0,256,(65,71,3),dtype=np.uint8))
        b=Image.fromarray(rng.integers(0,256,(65,71,3),dtype=np.uint8))
        p=predict(model,a,b,'cpu',32,1);q=predict(model,a,b,'cpu',32,4)
        self.assertTrue(np.allclose(p,q,atol=1e-6))
        self.assertEqual(q.shape,(65,71))

    def test_balanced_pseudo_gradients(self):
        x=torch.zeros(1,1,8,8,requires_grad=True);prob=torch.full_like(x,.01);prob[0,0,0,0]=.99
        loss,_=pseudo_loss(x,prob,class_balance=True);loss.backward()
        self.assertAlmostEqual(x.grad[0,0,0,0].item(),-.25,places=5)
        self.assertAlmostEqual(x.grad[0,0,1:,1:].sum().item(),.25*49/63,places=5)
        self.assertEqual(pseudo_loss(x,torch.full_like(x,.5),class_balance=True)[0].item(),0)

    def test_warmup_equivalence_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp);data=t/'data';self.fixture(data);manifests=t/'manifests';prepare(data,manifests,seeds=(17,))
            base=json.loads(Path('configs/base.json').read_text());base.update(manifest_dir=str(manifests),run_root=str(t/'direct'),epochs=2,steps_per_epoch=1,batch_size=2,crop=32,tile=32,device='cpu',cpu_threads=2,allow_fixture_data=True,warmup_epochs=3,validate_every=2)
            base['group']='A';a=train(base)
            base['group']='B';b=train(base)
            aa=torch.load(a/'last.pt',weights_only=False);bb=torch.load(b/'last.pt',weights_only=False)
            self.assertTrue(all(torch.equal(aa['model'][k],bb['model'][k]) for k in aa['model']))
            self.assertTrue(all(torch.equal(aa['teacher'][k],bb['teacher'][k]) for k in aa['teacher']))
            c=copy.deepcopy(base);c.update(group='A',run_root=str(t/'interrupted'))
            calls=0
            def interrupt(*args,**kwargs):
                nonlocal calls
                calls+=1
                if calls==2:raise InterruptedError('intentional')
                return evaluate(*args,**kwargs)
            with patch('cd.train.evaluate',side_effect=interrupt):
                with self.assertRaises(InterruptedError):train(c)
            c['resume']=True;resumed=train(c)
            rr=torch.load(resumed/'last.pt',weights_only=False)
            self.assertTrue(all(torch.equal(rr['model'][k],aa['model'][k]) for k in aa['model']))
            self.assertEqual(len((resumed/'history.jsonl').read_text().splitlines()),2)

if __name__=='__main__':unittest.main()
