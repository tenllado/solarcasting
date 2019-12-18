#!/usr/bin/env python

import pandas as pd
import datetime as dt

class McClear:
    def __init__(self, file):
        # Observation period;TOA;Clear sky GHI;Clear sky BHI;Clear sky DHI;Clear sky BNI
        cols = ["Op","TOA","GHI","BHI","DHI","BNI"]
        self.data = pd.read_csv(file, delimiter = ';', comment = '#', header=None, names = cols)
        self.data['Op'] = self.data['Op'].apply(lambda str: str.split('/')[0])
        self.data['Op'] = pd.to_datetime(self.data['Op'])
        self.data['Op'] = self.data['Op'].dt.tz_localize('UTC')

    def get_irradiance(self, date):
        date1 = date.replace(second = 0)
        date2 = date1 + dt.timedelta(0, 60)
        df = (self.data['Op'] >= date1) & (self.data['Op'] <= date2)
        df = self.data[df]['GHI']
        if df.size > 1:
            y1 = df.iloc[0]
            dy = df.iloc[1] - y1
            dx  = (date2 - date1).total_seconds()
            x_x1  = (date - date1).total_seconds()
            m   = dy / dx
            irr =  y1 + m * x_x1
        else:
            #considering constant
            irr = df.iloc[0]
		## mcclear gives info in Wh/m2 for every minute, while we use W/m2.
		## Thus, we need to multiply the value by 60 to obtain the power
        return 60*irr
