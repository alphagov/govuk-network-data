'''
Helper functions used in the EDA notebooks
'''

import numpy as np
import pandas as pd

def get_end_page(Page_List):
    return pd.eval(Page_List)[-1]

def get_end_page_event(Page_Event_List):          
    return pd.eval(Page_Event_List)[-1][-1]

def count_desktop(DeviceCategories):
    thelist  = pd.eval(DeviceCategories)
    desktop = 0
    mobile = 0
    other = 0
    for i in range(len(thelist)):
        if thelist[i][0] =='desktop':
            desktop = thelist[i][1]
        elif thelist[i][0] =='mobile':
            mobile = thelist[i][1]
        else:
            other = thelist[i][1]
    return desktop, mobile, other

def derive_new_variables(df):
    print("creating page sequence length vars")
    # string to list
    df['page_list_eval'] = df['Page_List'].map(pd.eval)
    # count list items in the page sequence (so this is page count for the journey)
    df['page_seq_len'] = df['page_list_eval'].map(len)

    # string to list
    df['page_list_NL_eval'] = df['Page_List_NL'].map(pd.eval)
    # Count the page sequence without loops so B -> A ->A is B -> A and length is 2
    df['page_seq_len_NL'] = df['page_list_NL_eval'].map(len)

    print("Creating search vars")

    # variable to count how many times do the keywords that identify search appear in the page sequence?
    df['count_search'] = df.PageSequence.str.count("/search?") + df.PageSequence.str.count("/search/")

    # new variable: does the event list include the term "start"? yes ->1, no ->0
    df['event_list_contains_start'] = np.where(df.Event_List.str.contains("start"), 1, 0)
    # new variable: does the page sequence include the term "start"? yes ->1, no ->0
    df['page_seq_contains_start'] = np.where(df.Sequence.str.contains("start"), 1, 0)
    # new variable: does the page sequence include the term "service.gov.uk"? yes ->1, no ->0
    # This identifies external links to a service which has passed a serivce assessment
    df['page_seq_contains_service.gov.uk'] = np.where(df.Sequence.str.contains("service.gov.uk"), 1, 0)

    df['final_page'] = df['Page_List'].map(get_end_page)
    df['final_interaction'] = df['Page_Event_List'].map(get_end_page_event)

    # new variable: does the page sequence include the terms which identify internal search? yes ->1, no ->0
    df['contains_search_regex'] = np.where(
        (df.PageSequence.str.contains("/search?")) | (df.PageSequence.str.contains("/search/")), 1, 0)

    df['contains_search_n'] = df['contains_search_regex'] * df['Page_Seq_Occurrences']

    df['desktop'], df['mobile'], df['other_device'] = zip(
        *df['DeviceCategories'].map(count_desktop))

    df['more_desktop'] = np.where(df['desktop'] > (df['mobile'] + df['other_device']), 1, 0)

    print("creating final_page_type")

    df['final_page_type'] = 'other'
    df.loc[df['final_page'].str.contains('/government/publications/'), 'final_page_type'] = 'government_publication'
    df.loc[df['final_page'].str.contains('log-in'), 'final_page_type'] = 'login'
    df.loc[df['final_page'].str.contains('sign-in'), 'final_page_type'] = 'login'
    df.loc[df['final_page'].str.contains('login'), 'final_page_type'] = 'login'
    df.loc[df['final_page'].str.contains('check'), 'final_page_type'] = 'check'
    df.loc[df['final_page'].str.contains('apply'), 'final_page_type'] = 'apply'
    df.loc[df['final_page'].str.contains('contact'), 'final_page_type'] = 'contact/enquiries'
    df.loc[df['final_page'].str.contains('enquiries'), 'final_page_type'] = 'contact/enquiries'
    df.loc[df['final_page'].str.contains(r'get-.*-information.*'), 'final_page_type'] = 'get_information'
    df.loc[df['final_page'].str.contains('send'), 'final_page_type'] = 'send'
    df.loc[df['final_page'].str.contains('find'), 'final_page_type'] = 'find'
    df.loc[df['final_page'].str.contains('calculat'), 'final_page_type'] = 'calculate/calculator'
    df.loc[df['final_page'].str.contains('order'), 'final_page_type'] = 'order'
    df.loc[df['final_page'].str.contains('manage'), 'final_page_type'] = 'manage'
    df.loc[df['final_page'].str.contains('update'), 'final_page_type'] = 'update'
    df.loc[df['final_page'].str.contains('eligibility'), 'final_page_type'] = 'eligibility'
    df.loc[df['final_page'].str.contains('estimate'), 'final_page_type'] = 'estimate'
    df.loc[df['final_page'].str.contains('renew'), 'final_page_type'] = 'renew'
    df.loc[df['final_page'].str.contains('pay'), 'final_page_type'] = 'pay'
    df.loc[df['final_page'].str.contains('claim'), 'final_page_type'] = 'claim'
    df.loc[df['final_page'].str.contains('change'), 'final_page_type'] = 'change'

    df['final_interaction_type'] = df.final_interaction.str.extract(r'<:<(.*)<:<', expand=False)
    df['final_external_link'] = df.final_interaction.str.extract(r'EVENT<:<External Link Clicked<:(.*)', expand=False)
    df['exit_to_assessed_service'] = np.where(df['final_external_link'].str.contains(r'.*service.gov.uk.*', na=False),
                                              1, 0)

    return df

def groupby_percent(df, groupby_var, unit_var, figsize=(10, 5)):
    x = df.groupby(groupby_var).count().reset_index()
    x['percent'] = 100*x[unit_var]/df.shape[0]
    x = x.sort_values(['percent'])

    s = pd.DataFrame(x[[groupby_var, unit_var,'percent']])

    return(s, x.plot(x=groupby_var, y='percent', kind='barh', figsize=figsize, color='#2B8CC4'))