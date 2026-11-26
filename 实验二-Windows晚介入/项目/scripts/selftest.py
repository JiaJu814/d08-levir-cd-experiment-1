"""Synthetic program checks only; does not open LEVIR or run real training."""
import copy
import importlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
import common
import supervisor
import report

class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='d08-selftest-');self.root=Path(self.tmp.name)
        self.results=self.root/'results';self.results.mkdir()
        self.saved=(supervisor.ROOT,supervisor.RESULTS,report.ROOT,report.RESULTS)
        supervisor.ROOT=self.root;supervisor.RESULTS=self.results;report.ROOT=self.root;report.RESULTS=self.results
    def tearDown(self):
        supervisor.ROOT,supervisor.RESULTS,report.ROOT,report.RESULTS=self.saved
        self.tmp.cleanup()
    def test_metric_defined_counts(self):
        m=common.metric(80,20,20,1000)
        self.assertEqual(m['f1'],0.8);self.assertAlmostEqual(m['iou'],2/3)
        self.assertAlmostEqual(m['false_positive_rate'],20/1020)
    def test_metric_empty_undefined(self):
        self.assertIsNone(common.metric(0,0,0,10)['f1'])
        self.assertEqual(common.metric(0,0,10,10)['f1'],0.)
    def test_atomic_json(self):
        p=self.root/'nested/a.json';common.write(p,{'a':1});common.write(p,{'a':2})
        self.assertEqual(common.read(p),{'a':2});self.assertFalse(p.with_name('a.json.tmp').exists())
    def test_zip_traversal_blocked(self):
        for name in ['../escape','/absolute','C:/escape','a\\b']:
            data=io.BytesIO()
            info=zipfile.ZipInfo(name);info.filename=name  # Keep an intentionally unsafe raw ZIP name on Windows.
            with zipfile.ZipFile(data,'w') as z:z.writestr(info,b'bad')
            data.seek(0)
            with zipfile.ZipFile(data) as z:
                with self.assertRaises(ValueError):list(common.safe_members(z))
    def test_missing_baseline_blocks_semi(self):
        cfg=common.read(common.ROOT/'config.json')
        self.assertFalse(supervisor.baseline_gate(cfg))
    def test_stable_baseline_gate(self):
        cfg=common.read(common.ROOT/'config.json')
        for k,f in [('sl10',.8),('sl100',.82)]:
            common.write(self.results/k/'result.json',dict(completed=True,best_val_f1=f))
            common.write(self.results/k/'history.json',[dict(val=dict(f1=f+x)) for x in [-.004,-.002,0]])
        self.assertTrue(supervisor.baseline_gate(cfg))
        common.write(self.results/'sl100/history.json',[dict(val=dict(f1=x)) for x in [.5,.6,.82]])
        self.assertFalse(supervisor.baseline_gate(cfg))
    def test_lock_rejects_duplicate(self):
        lock=supervisor.acquire()
        with self.assertRaises(RuntimeError):supervisor.acquire()
        lock.unlink()
    def test_partial_report_no_fake_gain(self):
        common.write(self.results/'task.json',dict(status='blocked',reason='synthetic test'))
        report.build()
        s=common.read(self.results/'summary.json')
        self.assertFalse(s['formal_goal_achieved']);self.assertIsNone(s['single_seed_validation_gain_pp'])
        self.assertTrue((self.root/'RETURN_TO_CHATGPT.zip').exists())
        with zipfile.ZipFile(self.root/'RETURN_TO_CHATGPT.zip') as z:
            self.assertIsNone(z.testzip());self.assertFalse(any(n.endswith('.pt') for n in z.namelist()))
    def test_real_deadline_guard_and_unrelated_process(self):
        import psutil
        scripts=self.root/'scripts';scripts.mkdir()
        for name in ['common.py','watchdog.py']:shutil.copy2(common.ROOT/'scripts'/name,scripts/name)
        (scripts/'report.py').write_text('from common import RESULTS\n(RESULTS/"guard_reported").touch()\n')
        common.write(self.root/'config.json',dict(report_reserve_seconds=2))
        common.write(self.results/'task.json',dict(status='running',deadline_epoch=time.time()+4))
        worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
        other=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
        common.write(self.results/'worker_identity.json',dict(pid=worker.pid,created=psutil.Process(worker.pid).create_time()))
        parent=psutil.Process()
        guard=subprocess.Popen([sys.executable,str(scripts/'watchdog.py'),'--parent',str(parent.pid),'--created',str(parent.create_time())])
        try:
            guard.wait(timeout=15)
            self.assertEqual(guard.returncode,0);worker.wait(timeout=3)
            self.assertIsNone(other.poll());self.assertTrue((self.results/'STOP').exists())
            self.assertEqual(common.read(self.results/'task.json')['reason'],'deadline')
            self.assertTrue((self.results/'guard_reported').exists())
        finally:
            for p in [guard,worker,other]:
                if p.poll() is None:p.terminate()
                p.wait(timeout=3)
    def test_official_model_cpu_shape_gradient_and_rng_isolation(self):
        import torch
        sys.path.insert(0,str(common.ROOT/'baseline'))
        from cd.model import ChangeDetector
        torch.set_num_threads(2);torch.manual_seed(17)
        m=ChangeDetector();a=torch.rand(2,3,32,32);b=torch.rand_like(a)
        y=m(a,b);self.assertEqual(tuple(y.shape),(2,1,32,32))
        y.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        state=torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):torch.manual_seed(100017);m(a,b)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

if __name__=='__main__':unittest.main(verbosity=2)
