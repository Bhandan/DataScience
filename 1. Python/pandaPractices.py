import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = pd.Series([1,3,5,np.nan,6,8])

#print(s)

dates = pd.date_range('20130101', periods=6)
#print(dates)

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
#print(df)

df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })
#print(df2)
#print(df2.dtypes)

#print(df.head)
#print(df.tail(6))
#print(df.index)
#print(df.columns)
#print(df.values)
#print(df.describe())
#print(df.T)
#print(df.sort_index(axis=1, ascending=False))
#print(df.sort_values(by='B'))
#print(df['A'])
#print(df[0:3])
#print(df.loc[dates[0]])
#print(df.loc[:,['A','B']])
#print(df.loc['20130102':'20130104',['A','B']])
#print(df.at[dates[0],'A'])
#print(df.iloc[3])
#print( df[df.A > 0])
#print(df[df > 0])
#print(df2 = df.copy())
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
#print(s1)
#print(df['F'] = s1)
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
#print(df1)
#print(df1.fillna(value=5))
pd.isnull(df1)
#print(df1)
#print(df.mean)
#print(df.mean(1))
s = pd.Series(np.random.randint(0, 7, size=10))
#print(s)
#print(s.value_counts())
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
#print(pd.merge(left, right, on='key'))
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
s = df.iloc[3]
#print(df.append(s, ignore_index=True))
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                    'B' : ['one', 'one', 'two', 'three',
                           'two', 'two', 'one', 'three'],
                       'C' : np.random.randn(8),
                      'D' : np.random.randn(8)})
#print(df)
#print(df.groupby('A').sum())
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),
                    'E' : np.random.randn(12)})
#print(df)
#print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
#print(ts.resample('5Min').sum())
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts_utc = ts.tz_localize('UTC')
#print(ts_utc.tz_convert('US/Eastern'))

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
#print(ts.plot())
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])

df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')
df.to_csv('foo.csv')
pd.read_csv('foo.csv')
#print( pd.read_csv('foo.csv'))
#df.to_hdf('foo.h5','df')
#print(pd.read_hdf('foo.h5','df'))
if pd.Series([False, True, False]):
  print("I was true")
  
                  





                     
                     
