
import pandas as pd


def summarized(partition, dim,qis):
    for qi in qis:
        partition = partition.sort_values(by=qi)
        if (partition[qi].iloc[0] !=partition[qi].iloc[-1]):
            s = f"[{partition[qi].iloc[0]} - {partition[qi].iloc[-1]}]"
            partition[qi] = [s]*partition[qi].size

    return partition

def anonymize(partition, ranks,k,qis):
    dim=ranks[0][0]

    partition = partition.sort_values(by=dim)
    si = partition[dim].count()
    mid = si//2

    lhs=partition[:mid]
    rhs=partition[mid:]

    if(len(lhs)>=k and len(rhs)>=k):
        return pd.concat([anonymize(lhs,ranks,k,qis), anonymize(rhs,ranks,k,qis)])
    
    return summarized(partition, dim,qis)
    

def mondrian(partition,qis,k):
    ranks={}

    for qi in qis:
        ranks[qi]=len(set(partition[qi]))

    #sort ranks
    ranks = sorted(ranks.items(), key=lambda t:t[1], reverse=True)

    #print(ranks)

    return anonymize(partition,ranks,k,qis)



