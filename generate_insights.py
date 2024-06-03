#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import json
import sys
import random
import datetime
from collections import defaultdict


def disease_insights():
    s = ['RECOMMENDED', 'CONSIDER', 'NULL']
    r = ["EXPECTED_INFECTIONS", 'OLD_INFECTIONS', 'RECENT_INFECTIONS']
    t = ['GROWTH', 'CONTACT']
    c = ['Potato', 'Maize']
    d = ['Grey Leaf Spot', 'Northern Corn Leaf Blight', 'Common Rust', 'Botrytis cinerea', 'Late Blight',
         'Early Blight']

    # random select suggestion, reason, treatment
    suggestion = random.choice(s)
    reason = random.choice(r)
    treatment = random.choice(t)
    crop = random.choice(c)
    if crop == 'Potato':
        disease = d[:3]
    if crop == 'Maize':
        disease = d[3:]

    if suggestion == 'CONSIDER':
        if reason == 'EXPECTED_INFECTIONS':
            if treatment == 'GROWTH':
                insight = 'Consider application of a fungicide which also protects new growth. Based on weather ' \
                          'information, the risk of infection is elevated for several consecutive days. For a ' \
                          'fast-growing crop, it’s advised to use a fungicide that also protects new leaves that have ' \
                          'yet to emerge.'
            if treatment == 'CONTACT':
                insight = 'Consider application of contact fungicide. There is an elevated chance the crop will be ' \
                          'infected soon. Please consider applying a contact fungicide to prevent an infection.'
        if reason == 'OLD_INFECTIONS':
            insight = 'Consider application of systemic fungicide. Your crop might be infected, ' \
                      'consider applying a systemic fungicide to suppress a recent infection from within the plant, ' \
                      'to prevent further damage and to prevent a new infection source from within your field.'

        if reason == 'RECENT_INFECTIONS':
            insight = 'Consider application of translaminar fungicide. Your crop might be infected, ' \
                      'consider applying a translaminar fungicide to suppress a recent infection. A translaminar ' \
                      'fungicide will prevent further penetration of the fungi in the plant.'
        

    if suggestion == 'RECOMMENDED':
        reason = random.choice(r)
        if reason == 'EXPECTED_INFECTIONS':
            if treatment == 'GROWTH':
                insight = 'Application of a fungicide that also protects new growth is required. Your crop is likely ' \
                          'infected, we strongly recommend to apply a translaminar fungicide to suppress ' \
                          'a recent infection. A translaminar fungicide will prevent further penetration of the fungi ' \
                          'in the plant.'
            if treatment == 'CONTACT':
                insight = 'Application of contact fungicide is required. Your crop is likely infected, ' \
                          'we strongly recommend to apply a translaminar fungicide to suppress a recent infection. A ' \
                          'translaminar fungicide will prevent further penetration of the fungi in the plant.'
        if reason == 'OLD_INFECTIONS':
            insight = 'Application of systemic fungicide is required. Your crop is likely infected, ' \
                      'we strongly recommend to apply a systemic fungicide to suppress a recent infection from within ' \
                      'the plant, to prevent further damage and to prevent a new infection source from within your ' \
                      'field.'

        if reason == 'RECENT_INFECTIONS':
            insight = 'Application of translaminar fungicide is required. Your crop is likely infected, ' \
                      'we strongly recommend to apply a translaminar fungicide to suppress a recent infection. A ' \
                      'translaminar fungicide will prevent further penetration of the fungi in the plant.'
        

    if suggestion == 'NULL':
        insight = 'Application of fungicide not necessary. The circumstances are favorable for a well protected crop.'
        spray_insight = ''

    return insight



def main():
    """
    """
    d_insights = []
    for i in range(3):
        d_insights.append(disease_insights())
    print(d_insights)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript terminated by the user.")
        sys.exit(1)
