"""仅验证软件行为，构造数据不进入任何课题结果。"""
import copy
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from cd.data import PairDataset
from cd.losses import supervised_loss, pseudo_loss, update_teacher
from cd.metrics import Metrics
from cd.model import ChangeDetector
from cd.official_siamunet_diff import SiamUnet_diff
from cd.prepare import prepare
from cd.utils import read_json

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metric_definitions(self):
        m=Metrics(0)
        m.update(np.array([[1,1],[0,0]]), np.array([[1,0],[1,0]]))
        r=m.result()
        self.assertEqual(r['f1'],0.5)
        self.assertEqual(r['iou'],1/3)
        self.assertEqual(r['false_positive_rate'],0.5)
        m=Metrics();m.update(np.zeros((8,8)),np.zeros((8,8)))
        self.assertIsNone(m.result()['f1'])
        self.assertIsNone(m.result()['boundary_f1'])

    def test_reference_network_equivalence(self):
        torch.manual_seed(1)
        reference=SiamUnet_diff(3,2).eval()
        ours=ChangeDetector().eval();ours.load_state_dict(reference.state_dict())
        a,b=torch.rand(1,3,33,35),torch.rand(1,3,33,35)
        with torch.no_grad():
            expected=reference(a,b).exp()[:,1:2]
            actual=ours(a,b).sigmoid()
        self.assertEqual(actual.shape,(1,1,33,35))
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6))

    def test_losses_and_ema(self):
        model=ChangeDetector();teacher=copy.deepcopy(model)
        x=torch.rand(2,3,32,32);y=(torch.rand(2,1,32,32)>.5).float()
        out=model(x,x)
        l=supervised_loss(out,y,2,1)
        no_pseudo,stats=pseudo_loss(out,torch.full_like(out,.5))
        self.assertEqual(no_pseudo.item(),0)
        self.assertEqual(stats['coverage'],0)
        loss=l+pseudo_loss(out,torch.full_like(out,.99),boundary_weight=2)[0]
        loss.backward()
        self.assertTrue(torch.isfinite(model.conv11.weight.grad).all())
        before=next(teacher.parameters()).clone()
        with torch.no_grad(): next(model.parameters()).add_(1)
        update_teacher(teacher,model,.9)
        self.assertTrue(torch.allclose(next(teacher.parameters()),before+.1,atol=1e-6))

    def test_split_and_hidden_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'dataset'; manifests=Path(tmp)/'manifests'
            rng=np.random.default_rng(55)
            for split,n in [('train',25),('val',2),('test',2)]:
                for folder in ['A','B','label']:(root/split/folder).mkdir(parents=True)
                for i in range(n):
                    for k in ('A','B'):
                        Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(root/split/k/f'{i}.png')
                    Image.fromarray((rng.random((32,32))>.8).astype(np.uint8)*255).save(root/split/'label'/f'{i}.png')
            prepare(root,manifests,seeds=(17,))
            s5=read_json(manifests/'17/5/labeled.json');s10=read_json(manifests/'17/10/labeled.json')
            self.assertEqual(len(s5),1)
            self.assertEqual(len(s10),2)
            self.assertEqual(read_json(manifests/'audit.json')['rounding'],'floor')
            self.assertTrue({r['a'] for r in s5}<={r['a'] for r in s10})
            u=read_json(manifests/'17/10/unlabeled.json')
            for r in u:self.assertNotIn('label',r)
            # 删除隐藏标签后无标签数据仍可加载，证明读取流程不依赖它们。
            for r in u:(root/'train/label'/r['name']).unlink()
            sample=PairDataset(u,32,True)[0]
            self.assertEqual(len(sample),4)
            self.assertEqual(sample[0].shape,(3,32,32))
            with self.assertRaises(ValueError):PairDataset(s10,32,True)

if __name__=='__main__':unittest.main()
