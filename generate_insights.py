#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import json
import sys
import math
import random
import datetime
from collections import defaultdict


def interval_extract(times):
	"""
	
	Param:
		times (list): Get interval of hours in times
	Return:
		insight (string): string with spray condtion insight
	"""
    length = len(times)
    i = 0
    while i < length:
        low = times[i]
        while i < length - 1 and times[i] + 1 == times[i + 1]:
            i += 1
        high = times[i]
        if (high - low) >= 1:
            yield [low, high]
        elif (high - low) == 1:
            yield [low, ]
            yield [high, ]
        else:
            yield [low, ]
        i += 1


def spray_condition_insight(parsed_dates, message):
	"""
	Create spray condition insights
	Param:
		parsed_dates (list): dates with spraying conditions matching message
		message (string): advice message to start output string
	Return:
		insight (string): string with spray condtion insight
	"""
    # Group by date
    date_groups = defaultdict(list)
    for dt in parsed_dates:
        date_groups[dt[:2]].append(int(dt[11:13]))

    # Construct the output string
    if message == 'very_good':
        output = "Spraying conditions are expected to be very good on the"
    if message == 'good':
        output = "Spraying conditions are expected to be good on the"
    if message == 'reasonable':
        output = "Spraying conditions are expected to be reasonable on the"

    date_strings = []
    for date, times in date_groups.items():
        continuous_intervals = interval_extract(times)
        for i in continuous_intervals:
            if len(i) == 1:
                date_strings.append(f"{output} {date[:2]}th of June around {i[0]}:00.")
            else:
                date_strings.append(
                    f"{output} {date[:2]}th of June between {i[0]}:00 and {i[1]}:00.")

    insight = ' '.join(date_strings)
    return insight


def get_spray_conditions():
	"""
	generating spray condition insights from input json files
	Return:
		insight (string): string with spray condtion insight
	"""
    f = open('spray.json')
    data = json.load(f)
    spray_times = []
    very_good = []
    good = []
    reasonable = []
    for i in data['content']:
        if data['content'][i]['advice'] == 'VERY_GOOD':
            very_good.append(datetime.datetime.fromtimestamp(int(i) / 1000).strftime('%d-%m-%Y %H:%M:%S'))
        if len(very_good) == 0:
            if data['content'][i]['advice'] == 'GOOD':
                good.append(datetime.datetime.fromtimestamp(int(i) / 1000).strftime('%d-%m-%Y %H:%M:%S'))
            if len(good) == 0:
                if data['content'][i]['advice'] == 'REASONABLE':
                    reasonable.append(datetime.datetime.fromtimestamp(int(i) / 1000).strftime('%d-%m-%Y %H:%M:%S'))
                if len(reasonable) == 0:
                    insight = 'There are not moments in the upcoming week when the spraying conditions are optimal'
                else:
                    insight = spray_condition_insight(reasonable, 'reasonable')
            else:
                insight = spray_condition_insight(good, 'good')
        else:
            insight = spray_condition_insight(very_good, 'very_good')
    return insight


def disease_insights():
	"""
	Create disease insights
	Return:
		insight (string): string with spray condtion insight
	"""
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
        spray_insight = get_spray_conditions()

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
        spray_insight = get_spray_conditions()

    if suggestion == 'NULL':
        insight = 'Application of fungicide not necessary. The circumstances are favorable for a well protected crop.'
        spray_insight = ''

    return insight + spray_insight


def irrigation_insights():
	"""
	Create irrigation insights
	Return:
		insight (string): string with irrigation insight
	"""
    s = ['REFILL', 'FULL', 'OPTIMAL']

    sws = random.choice(s)
    days_to_refill = random.randint(1,14)
    storage = random.randint(1,1) #check borders

    if sws == 'REFILL':
        insight = 'Soil water is now less than the refill point. Plant water stress can occur if refill status persists.'

    if sws == 'FULL':
        if days_to_refill > 7:
            insight = f'Soil water is above the full point. Plant damage and cost inefficiency can occur if full status persists.' \
            f'Without irrigation, soil water is expected to fall below the refill point in about {days_to_refill}'
        else:
            insight = f'Soil water is above the full point. Plant damage and cost inefficiency can occur if full status persists.' \
            f'Without irrigation, soil water is expected to fall below the refill point within the next {days_to_refill} days'

    if sws == 'OPTIMAL':
       if days_to_refill > 7:
           insight = f'Soil water is currently in the optimal range for plant health. Soil water is not expected to fall below the refill point within the next 7 days, even without irrigation.'  \
           f'There is currently storage capacity for up to {storage} inches of additional water'
       elif (days_to_refill - 1) < 2:
           insight = f'Soil water is currently in the optimal range for plant health. Irrigate in the next {days_to_refill-1} to maintain optimal range.' \
           f'There is currently storage capacity for up to {storage} inches of additional water'
       else:
           insight = f'Soil water is currently in the optimal range for plant health. Irrigate in the next day to avoid refill status' \
           f'There is currently storage capacity for up to {storage} inches of additional water'


    # time_refill = list()
    # full_run_time = list()
    # moisture_forecast = list()

    return insight


def main():
    """
    """
    # Irrigation insights
    i_insights = []
    for i in range(3):
        i_insights.append(irrigation_insights())

    for i in i_insights:
        print(i)

    # Disease insights
    # d_insights = []
    # for i in range(3):
    #     d_insights.append(disease_insights())
    # spray_insight = get_spray_conditions()
    # print(spray_insight)
    # for i in d_insights:
    #     print(i)




if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript terminated by the user.")
        sys.exit(1)
