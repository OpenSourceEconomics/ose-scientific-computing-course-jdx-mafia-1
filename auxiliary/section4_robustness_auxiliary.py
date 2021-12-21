def plots_prep():
    
    ## GDP MURDER FUNCTION TO CREATE PLOTS
    def gdp_murder_plotter(data,treat_unit,control_units,region_weights,title1,ax1):

        X3 = data.loc[data[time_identifier].isin(entire_period)]
        X3.index = X3.loc[:,unit_identifier]

        murd_treat_all   = np.array(X3.loc[(X3.index == treat_unit),('murd')]).reshape(1,len(entire_period))
        murd_control_all = np.array(X3.loc[(X3.index.isin(control_units)),('murd')]).reshape(len(control_units),len(entire_period))
        gdp_control_all  = np.array(X3.loc[(X3.index.isin(control_units)),('gdppercap')]).reshape(len(control_units),len(entire_period))
        gdp_treat_all    = np.array(X3.loc[(X3.index == treat_unit),('gdppercap')]).reshape(1,len(entire_period))

        synth_murd = region_weights.T @ murd_control_all

        synth_gdp = region_weights.T @ gdp_control_all

        diff_GDP = (((gdp_treat_all-synth_gdp)/(synth_gdp))*100).ravel()
        diff_murder = (murd_treat_all - synth_murd).ravel()

        diff_data = pd.DataFrame({'Murder Gap':diff_murder,
                                 'GDP Gap': diff_GDP},
                                 index=data.year.unique())

        year = diff_data.index.values
        ax1.bar(year,diff_data['GDP Gap'],width = 0.5,label = 'GDP per capita')
        ax1.axhline(0)
        ax1.title.set_text(title1)
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.plot(diff_data['Murder Gap'],color='black',label = 'Murders')
        ax2.axhline(0)
        ax2.tick_params(axis='y')

        plt.axvspan(1975, 1980, color='y', alpha=0.5, lw=0,label='Mafia Outbreak')
        ax1.set_ylim(-30,30)
        ax2.set_ylim(-4,4)
    
    # SETTINGS
    unit_identifier = 'reg'
    time_identifier = 'year'
    matching_period = list(range(1951, 1961))
    control_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
    outcome_variable = ['gdppercap']
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
    entire_period = list(range(1951, 2008))
    reps=1

    
    

    

def multiplot():
    
    # Plots graphs in 3x3 format
    fig, fig_axes = plt.subplots(ncols=3, nrows=3,figsize=(10,10))

    
    # Only Apulia in treatment group: Changes treat_unit to region number 16
    treat_unit = 16
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,
                        control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(15,1)
    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(a) Only Apulia in treatment group',fig_axes[0,0])

    
    # Only Basilicata in treatment group: Changes treat_unit to region number 17
    treat_unit = 17
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(15,1)
    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(b) Only Basilicata in treatment group',fig_axes[0,1])

    
    # No Molise in control group: Removes region 14 from control_unit
    treat_unit = 21
    control_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20]
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,treat_unit,
                        control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(14,1)
    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(c) No Molise in control group',fig_axes[0,2])


    # No Abruzzo in control group: Removes region 13 from control_unit
    control_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 20]
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(14,1)
    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(d) No Abruzzo in control group',fig_axes[1,0])


    # No Sardinia in control group: Removes region 20 from control_unit
    control_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(14,1)
    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(e) No Sardinia in control group',fig_axes[1,1])


    # Include crimes in predictor variables: add variable 'robkidext' in predictor_variables
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density','robkidext']
    control_units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(15,1)
    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(f) Include crimes in predictor variables',fig_axes[1,2])


    # Match over 1951 to 1975: change matching_period from (1951,1961) to (1951, 1976)
    predictor_variables = ['gdppercap', 'invrate', 'shvain', 'shvaag', 'shvams', 'shvanms', 'shskill', 'density']
    matching_period = list(range(1951, 1976))
    output_object = SCM(data,unit_identifier,time_identifier,matching_period,
                        treat_unit,control_units,outcome_variable,predictor_variables,reps)
    region_weights = output_object[1].reshape(15,1)
    gdp_murder_plotter(data,treat_unit,control_units,region_weights,'(g) Matching period 1951-1975',fig_axes[2,0])


    plt.tight_layout()
    plt.show()