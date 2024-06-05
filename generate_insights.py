#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import json
import sys
import csv
import random
import datetime
from collections import defaultdict


# import re
# import nltk
# from gensim.summarization import summarize
# nltk.download('punkt')


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


# def get_summary_spray_conditions(text):
#     # Split the text at each period followed by a space to create a new line
#     # print(text)
#     formatted_text = text.replace('. ', '.\n')
#     # print(formatted_text)
#     # print('i')
#     # Wrap the formatted text with triple double quotes
#     # triple_quoted_text = f'"""{formatted_text}"""'
#     # print(triple_quoted_text)
#
#
#     summary = summarize(formatted_text)
#     # print(type(summary))
#     print(summary)
#     # summary returns empty strings
#     return summary


def spray_condition_insight(parsed_dates, message):
    """
    Create spray condition insights
    Param:
        parsed_dates (list): dates with spraying conditions matching message
        message (string): advice message to start output string
    Return:
        insight (string): string with spray condition insight
    """
    # Group by date
    date_groups = defaultdict(list)
    for dt in parsed_dates:
        date_groups[dt[:5]].append(int(dt[11:13]))

    # Construct the output string
    if message == 'very_good':
        output = "Spraying conditions are expected to be very good on the"
    if message == 'good':
        output = "Spraying conditions are expected to be good on the"
    if message == 'reasonable':
        output = "Spraying conditions are expected to be reasonable on the"
    input_strings = [output]

    date_strings = []
    for date, times in date_groups.items():
        continuous_intervals = interval_extract(times)
        date_obj = datetime.datetime.strptime(date, "%d-%m")
        date_text = date_obj.strftime("%dth of %B")
        for i in continuous_intervals:
            if len(i) == 1:
                date_strings.append(f"{output} {date_text} around {i[0]}:00.")
                input_strings.append(f"{date_text} around {i[0]}:00")
            else:
                date_strings.append(
                    f"{output} {date_text} between {i[0]}:00 and {i[1]}:00.")
                input_strings.append(f"{date_text} between {i[0]}:00 and {i[1]}:00.")
    input = ' '.join(input_strings)
    insight = ' '.join(date_strings)
    return insight, input


def get_spray_conditions():
    """
    generating spray condition insights from input json files
    Return:
        insight (string): string with spray condition insight
    """

    f = open('spray.json')
    data = json.load(f)
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
                    insight, input = spray_condition_insight(reasonable, 'reasonable')
            else:
                insight, input = spray_condition_insight(good, 'good')
        else:
            insight, input = spray_condition_insight(very_good, 'very_good')
    return insight, input


def disease_insights():
    """
    Create disease insights
    Return:
        insight (string): string with spray condition insight
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
    spray_input = ''
    spray_insight = ''
    input = ''

    if suggestion == 'CONSIDER':
        input += 'Suggestion is to consider application of a fungicide.'
        if reason == 'EXPECTED_INFECTIONS':
            input += 'The reason is expected infections. '
            if treatment == 'GROWTH':
                insight = 'Consider application of a fungicide which also protects new growth. Based on weather ' \
                          'information, the risk of infection is elevated for several consecutive days. For a ' \
                          'fast-growing crop, it’s advised to use a fungicide that also protects new leaves that have ' \
                          'yet to emerge.'
                input += 'The treatment is to apply a fungicide that also protects new growth. '
            if treatment == 'CONTACT':
                insight = 'Consider application of contact fungicide. There is an elevated chance the crop will be ' \
                          'infected soon. Please consider applying a contact fungicide to prevent an infection.'
                input += 'The treatment is to apply a contact fungicide. '
        if reason == 'OLD_INFECTIONS':
            insight = 'Consider application of systemic fungicide. Your crop might be infected, ' \
                      'consider applying a systemic fungicide to suppress a recent infection from within the plant, ' \
                      'to prevent further damage and to prevent a new infection source from within your field.'
            input += 'The reason is old infections. The treatment is to apply a systemic fungicide.'
        if reason == 'RECENT_INFECTIONS':
            insight = 'Consider application of translaminar fungicide. Your crop might be infected, ' \
                      'consider applying a translaminar fungicide to suppress a recent infection. A translaminar ' \
                      'fungicide will prevent further penetration of the fungi in the plant.'
            input += 'The reason is recent infections. The treatment is to apply a translaminar fungicide.'
        spray_insight, spray_input = get_spray_conditions()

    if suggestion == 'RECOMMENDED':
        input += 'The suggestion is application of a fungicide is required. '
        if reason == 'EXPECTED_INFECTIONS':
            input += 'The reason is expected infections. '
            if treatment == 'GROWTH':
                insight = 'Application of a fungicide that also protects new growth is required. Your crop is likely ' \
                          'infected, we strongly recommend to apply a translaminar fungicide to suppress ' \
                          'a recent infection. A translaminar fungicide will prevent further penetration of the fungi ' \
                          'in the plant.'
                input += 'The treatment is to apply a fungicide that also protects new growth. '
            if treatment == 'CONTACT':
                insight = 'Application of contact fungicide is required. Your crop is likely infected, ' \
                          'we strongly recommend to apply a translaminar fungicide to suppress a recent infection. A ' \
                          'translaminar fungicide will prevent further penetration of the fungi in the plant.'
                input += 'The treatment is to apply a contact fungicide.'
        if reason == 'OLD_INFECTIONS':
            insight = 'Application of systemic fungicide is required. Your crop is likely infected, ' \
                      'we strongly recommend to apply a systemic fungicide to suppress a recent infection from within ' \
                      'the plant, to prevent further damage and to prevent a new infection source from within your ' \
                      'field. '
            input += 'The reason is old infections. The treatment is to apply a systemic fungicide.'
        if reason == 'RECENT_INFECTIONS':
            insight = 'Application of translaminar fungicide is required. Your crop is likely infected, ' \
                      'we strongly recommend to apply a translaminar fungicide to suppress a recent infection. A ' \
                      'translaminar fungicide will prevent further penetration of the fungi in the plant.'
            input += 'The reason is recent infections. The treatment is to apply a translaminar fungicide.'
        spray_insight, spray_input = get_spray_conditions()

    if suggestion == 'NULL':
        input = 'No suggestions.'
        insight = 'Application of fungicide not necessary. The circumstances are favorable for a well protected crop.'
        spray_insight = ''

    d_insight = insight + ' ' + spray_input
    d_input = input + ' ' + spray_insight
    return d_insight, d_input


def irrigation_insights():
    """
    Create irrigation insights
    Return:
        insight (string): string with irrigation insight
    """

    chance_rain = random.randint(0, 100)

    # if chance_rain > 0:
    #     daily_precipitation = 0
    # else:
    #     daily_precipitation = round(random.uniform(0.00, 5.00), 2)

    s = ['REFILL', 'FULL', 'OPTIMAL']

    sws = random.choice(s)
    days_to_refill = random.randint(1, 14)
    storage = round(random.uniform(0.00, 10.00), 2)
    water_needed = round(random.uniform(0.00, 5.00), 2)
    input = ''

    if sws == 'REFILL':
        insight = 'Soil water is now less than the refill point. Plant water stress can occur if refill status persists. '
        input += 'Soil water status is refill. '
        input += f'{water_needed} inches of water is needed to achieve optimal soil water status. '
        if chance_rain <= 10:
            insight += f'No good chances of rain, consider irrigation.'
        elif 10 < chance_rain < 60:
            insight += f"Chances of rain in the next few days. If it does not rain, consider irrigation"
        elif chance_rain >= 60:
            insight += "Great chance of rain. Hold off on irrigation"
    if sws == 'FULL':
        input += 'Soil water status is full. '
        if days_to_refill > 7:
            insight = f'Soil water is above the full point. Plant damage and cost inefficiency can occur if full status persists. ' \
                      f'Without irrigation, soil water is expected to fall below the refill point in about {days_to_refill} days. '
            input += f'Soil water is expected to fall below refill point in about {days_to_refill} days. '
        else:
            insight = f'Soil water is above the full point. Plant damage and cost inefficiency can occur if full status persists. ' \
                      f'Without irrigation, soil water is expected to fall below the refill point within the next {days_to_refill} days. '
            input += f'Days until refill status is reached is {days_to_refill} days. '
    if sws == 'OPTIMAL':
        input += 'Soil water status is optimal. '
        if days_to_refill > 7:
            insight = f'Soil water is currently in the optimal range for plant health. Soil water is not expected to fall below the refill point within the next 7 days, even without irrigation. ' \
                      f'There is currently storage capacity for up to {storage} inches of additional water. '
            input += f'Days until refill status is reached is {days_to_refill} days. '
        elif (days_to_refill - 1) < 2:
            insight = f'Soil water is currently in the optimal range for plant health. Irrigate in the next {days_to_refill - 1} to maintain optimal range. ' \
                      f'There is currently storage capacity for up to {storage} inches of additional water. '
            input += f'Days until refill status is reached is {days_to_refill} days. '
        else:
            insight = f'Soil water is currently in the optimal range for plant health. Irrigate in the next day to avoid refill status' \
                      f'There is currently storage capacity for up to {storage} inches of additional water. '
            input += f'Days until refill status is reached is {days_to_refill}. '

        if chance_rain <= 10:
            insight += f'No good chances of rain, consider irrigation.'
        elif 10 < chance_rain < 60:
            insight += f"Chances of rain in the next few days. If it does not rain, consider irrigation"
        elif chance_rain >= 60:
            insight += "Great chance of rain. Hold off on irrigation"

    input += f'{chance_rain}% chance of rain.'

    return insight, input


def main():
    """
    """
    i_insights, d_insights, i_inputs, d_inputs = ([] for i in range(4))

    for i in range(5000):
        # irrigation insights
        i_insight, i_input = irrigation_insights()
        i_insights.append(i_insight)
        i_inputs.append(i_input)

        # disease insights
        d_insight, d_input = disease_insights()
        d_insights.append(d_insight)
        d_inputs.append(d_input)

    with open('data_small.csv', 'w') as f:
        # using csv.writer method from CSV package
        writer = csv.writer(f)
        writer.writerow(['input', 'output'])
        for i in range(len(i_inputs)):
            input_seq = i_inputs[i] + ' ' + d_inputs[i]
            summary_seq = i_insights[i] + ' ' + d_insights[i]
            writer.writerow([input_seq, summary_seq])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript terminated by the user.")
        sys.exit(1)

