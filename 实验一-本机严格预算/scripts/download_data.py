"""来自LEVIR-CD作者页面公开Drive目录；仅供学术研究。"""
import hashlib
import json
import zipfile
from pathlib import Path
import urllib.request
import urllib.parse
import http.cookiejar
from html.parser import HTMLParser
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
# ID核对自作者公开目录，不使用LEVIR-CD+或第三方重划分版本。
FILES = {
    'val': '1BqSt4ueO7XAyQ_84mUjswUSJt13ZBuzG',
    'train': '18GuoCuBn48oZKAlEo-LrNwABrFhVALU-',
    'test': '1jj3qJD_grJlgIhUWO09zibRGJe0R4Tn0',
}

class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__();self.action=None;self.fields={};self.active=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='form' and a.get('id')=='download-form':
            self.action=a.get('action');self.active=True
        if tag=='input' and self.active and 'name' in a:
            self.fields[a['name']]=a.get('value','')
    def handle_endtag(self,tag):
        if tag=='form':self.active=False

def download(file_id, destination):
    # 下载中断后使用HTTP Range续传，保留已取得的数据。
    opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    initial='https://drive.google.com/uc?'+urllib.parse.urlencode({'id':file_id,'export':'download'})
    last_error=None
    for retry in range(6):
        try:
            if destination.exists() and zipfile.is_zipfile(destination):
                return str(destination)
            offset=destination.stat().st_size if destination.exists() else 0
            url=initial
            for confirmation in range(4):
                request=urllib.request.Request(url,headers={'Range':'bytes='+str(offset)+'-'} if offset else {})
                with opener.open(request,timeout=45) as response:
                    if 'text/html' in response.headers.get('Content-Type',''):
                        form=DownloadForm();form.feed(response.read().decode('utf-8'))
                        if not form.action:raise RuntimeError('Google Drive未提供下载表单，可能是配额或权限限制。')
                        url=form.action+'?'+urllib.parse.urlencode(form.fields)
                        continue
                    if offset:
                        expected='bytes '+str(offset)+'-'
                        if response.status!=206 or not response.headers.get('Content-Range','').startswith(expected):
                            raise RuntimeError('服务器未确认Range续传；保留part文件，拒绝覆盖已下载内容。')
                    total=offset;last=time.time()
                    print('transfer',destination.name,'starting at',offset,flush=True)
                    with open(destination,'ab' if offset else 'wb') as target:
                        while True:
                            chunk=response.read(1024*1024)
                            if not chunk:break
                            target.write(chunk);total+=len(chunk)
                            if time.time()-last>10:
                                target.flush();print(destination.name,round(total/1024**2,1),'MiB',flush=True);last=time.time()
                    print('downloaded',destination,total,flush=True)
                    return str(destination)
            raise RuntimeError('多次确认后仍未取得ZIP。')
        except Exception as exc:
            last_error=exc
            print('download attempt',retry+1,'failed:',repr(exc),'will retain partial file',flush=True)
            if retry<5:time.sleep(min(10,2**retry))
    raise RuntimeError('下载多次失败，已保留part文件供续传。') from last_error

def unpack(archive, split):
    dest = ROOT/'data/LEVIR-CD'/split
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            parts = list(Path(info.filename).parts)
            if not parts or parts[0]=='__MACOSX' or any(p.startswith('.') for p in parts):
                continue
            if parts[0]==split:
                parts=parts[1:]
            if not parts or parts[0] not in ('A','B','label'):
                continue
            out=dest.joinpath(*parts).resolve()
            if not out.is_relative_to(dest.resolve()):
                raise ValueError('不安全的压缩包路径')
            if info.is_dir():
                out.mkdir(parents=True,exist_ok=True)
            else:
                out.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src, out.open('wb') as target:
                    import shutil
                    shutil.copyfileobj(src,target)

def main():
    download_dir=ROOT/'data/downloads'; download_dir.mkdir(parents=True,exist_ok=True)
    status={}
    for split,file_id in FILES.items():
        if file_id is None:
            raise RuntimeError('尚未核实'+split+'下载ID。')
        archive=download_dir/(split+'.zip')
        if not archive.exists():
            result=download(file_id, Path(str(archive)+'.part'))
            if result is None or not zipfile.is_zipfile(result):
                raise RuntimeError('下载失败或不是有效ZIP: '+split)
            Path(result).rename(archive)
        if not zipfile.is_zipfile(archive):
            raise ValueError('无效ZIP: '+str(archive))
        unpack(archive,split)
        sha=hashlib.sha256()
        with archive.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):sha.update(chunk)
        status[split]={'file_id':file_id,'sha256':sha.hexdigest(),'bytes':archive.stat().st_size}
        (download_dir/'download_status.json').write_text(json.dumps(status,indent=2))

if __name__=='__main__':
    (ROOT/'work/downloader.pid').write_text(str(__import__('os').getpid()))
    main()
