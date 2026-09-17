import argparse, json
from .pipeline import run

def main():
    p=argparse.ArgumentParser(description="PixOrb SIH26166 lunar image correspondence")
    p.add_argument('--reference',required=True); p.add_argument('--source',required=True); p.add_argument('--output',default='results/run')
    p.add_argument('--model',choices=['homography','affine'],default='homography'); p.add_argument('--threshold',type=float,default=3.0)
    a=p.parse_args(); r=run(a.reference,a.source,a.output,a.model,a.threshold)
    print(json.dumps(r['metrics'],indent=2,default=str))
if __name__=='__main__': main()
