"""Academic corpus classification using both headline and content."""
import argparse,json,re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix

def preprocess(text):
    text=re.sub(r'https?://\S+',' ',str(text).lower())
    return re.sub(r'\s+',' ',re.sub(r'[^a-z\s]',' ',text)).strip()

def prepare(fake,true):
    for frame in (fake,true):
        if not {'title','text'}<=set(frame.columns):raise ValueError('Expected title and text columns.')
    frame=pd.concat([fake.assign(label=0),true.assign(label=1)],ignore_index=True)
    frame['document']=(frame.title.fillna('')+' '+frame.text.fillna('')).map(preprocess)
    frame=frame[frame.document.str.len()>0]
    conflicts=frame.groupby('document').label.nunique()
    frame=frame[~frame.document.isin(conflicts[conflicts>1].index)].drop_duplicates('document')
    return frame

def train_evaluate(fake,true):
    frame=prepare(fake,true)
    train,test=train_test_split(frame,test_size=.2,stratify=frame.label,random_state=42)
    assert not set(train.document)&set(test.document)
    model=Pipeline([('tfidf',TfidfVectorizer(max_features=30000,stop_words='english',ngram_range=(1,2))),('svm',LinearSVC(random_state=42))])
    model.fit(train.document,train.label);pred=model.predict(test.document)
    return {'dataset':'Supplied Fake/True news corpus','seed':42,'train_rows':len(train),'test_rows':len(test),
            'uses_title_and_content':True,'train_test_text_overlap':0,
            'accuracy':float(accuracy_score(test.label,pred)),'macro_f1':float(f1_score(test.label,pred,average='macro')),
            'classification_report':classification_report(test.label,pred,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(test.label,pred,labels=[0,1]).tolist(),
            'limitation':'Random split within one collected corpus; source and writing-style shortcuts remain possible. Not a truth verification system.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--fake',type=Path,required=True);p.add_argument('--true',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('metrics.json'))
    a=p.parse_args();r=train_evaluate(pd.read_csv(a.fake),pd.read_csv(a.true));a.output.write_text(json.dumps(r,indent=2),encoding='utf-8');print(r['accuracy'],r['macro_f1'])
if __name__=='__main__':main()
