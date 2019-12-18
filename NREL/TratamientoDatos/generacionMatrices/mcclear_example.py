#!/usr/bin/env python

import mcclear as mc
import datetime as dt

if __name__=="__main__":
    date = dt.datetime(2011, 12, 31, 23, 45, 0)
    m = mc.McClear('mcclear-dhhl6-2010-2011.csv')
    print(m.data)
    irr = m.get_irradiance(date)
    print(irr)
