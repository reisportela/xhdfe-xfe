"""Small exact-rational references, independent of xhdfe and floating solvers."""
from fractions import Fraction as F


def invert(matrix):
    n=len(matrix)
    work=[list(row)+[F(i==j) for j in range(n)] for i,row in enumerate(matrix)]
    for column in range(n):
        pivot=next(row for row in range(column,n) if work[row][column])
        work[column],work[pivot]=work[pivot],work[column]
        diagonal=work[column][column]
        work[column]=[value/diagonal for value in work[column]]
        for row in range(n):
            if row==column:continue
            multiple=work[row][column]
            work[row]=[a-multiple*b for a,b in zip(work[row],work[column])]
    result=[row[n:] for row in work]
    assert all(sum(matrix[i][k]*result[k][j] for k in range(n))==F(i==j)
               for i in range(n) for j in range(n))
    return result


def fit(design,response,weights):
    design=[[F(float(value)) for value in row] for row in design]
    response=[F(float(value)) for value in response];weights=[F(float(value)) for value in weights]
    n=len(design);p=len(design[0]);assert n>p and all(w>0 for w in weights)
    gram=[[sum(w*row[j]*row[k] for w,row in zip(weights,design)) for k in range(p)] for j in range(p)]
    rhs=[sum(w*row[j]*value for w,row,value in zip(weights,design,response)) for j in range(p)]
    bread=invert(gram)
    coefficients=[sum(a*b for a,b in zip(row,rhs)) for row in bread]
    residual=[value-sum(a*b for a,b in zip(row,coefficients)) for row,value in zip(design,response)]
    assert all(sum(w*row[j]*u for w,row,u in zip(weights,design,residual))==0 for j in range(p))
    rss=sum(w*u*u for w,u in zip(weights,residual))
    variance=rss/F(n-p)*bread[-1][-1]
    score=[w*u*sum(a*b for a,b in zip(bread[-1],row)) for w,u,row in zip(weights,residual,design)]
    robust=F(n,n-p)*sum(value*value for value in score)
    return dict(coefficients=coefficients,residuals=residual,variance=variance,robust_variance=robust)
