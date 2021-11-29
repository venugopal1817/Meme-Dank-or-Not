#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:39:53 2021

@author: saurabh
"""

import predict
import analysis
import streamlit as st
pg = {
    "Predict": predict
}
st.sidebar.title('Navigation')
select = st.sidebar.radio("Go to", list(pg.keys()))
pg = PAGES[select]
pg.main()