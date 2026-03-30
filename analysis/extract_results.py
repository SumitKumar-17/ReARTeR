import json

files = [
    (50,  'scale_50_20260330_165129.json'),
    (100, 'scale_100_20260330_165443.json'),
    (300, 'scale_300_20260330_170423.json'),
    (500, 'scale_500_20260330_172006.json'),
]

for scale, fname in files:
    with open(f'/home/rtx3060/ReARTeR/analysis/results/{fname}') as f:
        data = json.load(f)
    print(f'\n=== {scale}Q ===')
    for r in data['results']:
        m = r['metrics']
        em   = round(m.get('em',0),4)
        f1   = round(m.get('f1',0),4)
        acc  = round(m.get('acc',0),4)
        prec = round(m.get('precision',0),4)
        rec  = round(m.get('recall',0),4)
        t    = r['elapsed']
        print(f"  {r['method']:<20} EM={em}  F1={f1}  Acc={acc}  Prec={prec}  Rec={rec}  time={t}s")
    # Sample predictions from each method
    print('  -- Sample Q&A:')
    zs = next(r for r in data['results'] if r['method']=='zero_shot')
    nr = next(r for r in data['results'] if r['method']=='naive_rag')
    ra = next(r for r in data['results'] if r['method']=='rearter')
    for i in range(5):
        q   = zs['items'][i]['question'][:60]
        g   = zs['items'][i]['golden'][0]
        p0  = str(zs['items'][i].get('pred',''))[:35].replace('\n',' ')
        p1  = str(nr['items'][i].get('pred',''))[:35].replace('\n',' ')
        p2  = str(ra['items'][i].get('pred',''))[:35].replace('\n',' ')
        st  = ra['items'][i].get('steps','?')
        print(f'     [{i+1}] {q}')
        print(f'          Gold={g} | ZS={p0} | RAG={p1} | ReARTeR({st}steps)={p2}')
