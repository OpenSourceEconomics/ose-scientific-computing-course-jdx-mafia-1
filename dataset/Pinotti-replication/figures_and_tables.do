
capture log close
clear all
set mem 200m
set matsize 800
set more off



local dir "C:\" // insert your directory here

cd "`dir'"




*********************************** DESCRIPTIVE FIGURES AND TABLES ************************************
use dataset, clear
keep if year>=1983

***************** Figure 1 ******************
collapse gdppercap mafia murd ext fire kidnap rob smug drug theft orgcrime, by(region reg)
twoway (scatter gdppercap mafia if reg==15 | reg==18 | reg==19, mlabel(region) msymbol(triangle) mcolor(black) mlabp(1) mlabc(black)) (scatter gdppercap mafia if reg==16 | reg==17, mlabel(region) msymbol(triangle_hollow) mcolor(black) mlabp(1) mlabc(black)) (scatter gdppercap mafia if reg<=12 | reg==20, mcolor(black) msymbol(circle_hollow)), xtitle(, margin(medsmall)) ytitle(, margin(medsmall)) legend(off) 
graph save "`dir'\results\Figure1", replace

***************** Figure 2 ******************
sort reg
log using "`dir'\results\data_for_Figure2.log", replace
list reg region mafia if reg<=20
log close


***************** Figure 3 ******************
twoway (scatter murd mafia if reg==15 | reg==18 | reg==19, mlabel(region) msymbol(triangle) mcolor(black) mlabp(1) mlabc(black)) (scatter murd mafia if reg==16 | reg==17, mlabel(region) msymbol(triangle_hollow) mcolor(black) mlabp(1) mlabc(black)) (scatter murd mafia if reg<=12  | reg==20, mcolor(black) msymbol(circle_hollow)), xtitle(, margin(medsmall)) ytitle(, margin(medsmall)) legend(off) 
graph save "`dir'\results\Figure3", replace

***************** Appendix Figure A1 ******************
foreach var in ext fire kidnap rob smug drug theft orgcrime {
twoway (scatter `var' mafia if reg==15 | reg==18 | reg==19, mlabel(region) msymbol(triangle) mcolor(black) mlabp(1) mlabc(black)) (scatter `var' mafia if reg==16 | reg==17, mlabel(region) msymbol(triangle_hollow) mcolor(black) mlabp(1) mlabc(black)) (scatter `var' mafia if reg<=12  | reg==20, mcolor(black) msymbol(circle_hollow)), xtitle(, margin(medsmall)) ytitle(, margin(medsmall)) legend(off) xtitle("`var'")
graph copy gph`var', replace
}
graph combine gphext gphfire gphkidnap gphrob gphsmug gphdrug gphtheft gphorgcrime, xsize(8) ysize(10) cols(2) scale(.7)
graph save "`dir'\results\AppxFigureA1", replace


***************** Figure 4 ******************
use dataset, clear
keep if reg>20
keep murd year region
reshape wide murd, i(year) j(region) string
twoway (connected murdNEW murdHIS murdSTH murdNTH year)
graph save "`dir'\results\Figure4", replace


***************** Figure 5 ******************
use dataset, clear 
gen decade="1960-65" if year>=1960&year<=1965
replace decade="1965-70" if year>1965&year<=1970
replace decade="1970-75" if year>=1970&year<=1975
replace decade="1975-80" if year>=1975&year<=1980
replace decade="1980-85" if year>=1980&year<=1985
replace decade="1985-90" if year>=1985&year<=1990
replace decade="1990-95" if year>=1990&year<=1995

keep if reg>20
drop if decade==""
tsset reg year 
gen gdpgwt=(gdppercap-L.gdppercap)/L.gdppercap
collapse gdpgwt, by(region decade)
reshape wide gdpgwt, i(decade) j(region) string
log using "`dir'\results\data_for_Figure5.log", replace
order gdpgwtNEW gdpgwtHIS gdpgwtOTH
list 
log close





************************************ ECONOMETRIC ANALYSIS ************************************
use dataset, clear 
tsset reg year 

synth gdppercap gdppercap invrate shvain shvaag shvams shvanms shskill density, trunit(21) trperiod(1975) xperiod(1951(1)1960) mspeperiod(1951(1)1960) counit(1 2 3 4 5 6 7 8 9 10 11 12 13 14 20) nested fig

mat weights=e(W_weights)
mat weights=weights[1..15,2]

******** Table 1 **********
log using "`dir'\results\Table1.log", replace
* first 2 columns
mat list e(X_balance)
* last 4 columns
tabstat gdppercap invrate shvain shvaag shvams shvanms shskill density if year<=1960&(reg<15|reg==20), c(s) st(me sd min max)
log close 


******** Figure 6 and 7 **********
use dataset, clear
keep if reg<15|reg==20|reg==21
sort reg
keep gdppercap murd reg year
reshape wide gdppercap murd, i(year) j(reg)
rename gdppercap21 treatgdp
rename murd21 treatmurd
mkmat gdppercap*, matrix(synthgdp)
mat synthgdp=synthgdp*weights
svmat synthgdp
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*weights
svmat synthmurd
gen gapgdp=(treatgdp-synthgdp)/synthgdp*100
gen gapmurd=treatmurd-synthmurd


twoway (line synthgdp year, lcolor(gs10) lpattern(dash)) (line treatgdp year, lwidth(medthick) lcolor(black)) ,  ///
	xlabel(1955(10)2005) xtitle("") ///
	ytitle("GDP per capita, constant 1990 euros", margin(medium)) legend(region(lcolor(none)) order(2 "actual with mafia" 1 "synthetic control") bcolor(none) cols(2))
graph save "`dir'\results\Figure6", replace


gen shadeup=20 if year>=1975 & year<=1980
gen shadedown=-20 if year>=1975 & year<=1980
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapgdp year, bcolor(gs10) fintensity(100)) (line gapmurd year, yaxis(2) lcolor(black)), ///
	ytitle("GDP per capita, % gap", margin(medium)) ytitle("murder rate, difference", axis(2) margin(medium)) ///
	ylabel(-4(2)4, axis(2)) xlabel(1955(10)2005) xtitle("") legend(region(lcolor(none)) order(1 "mafia outbreak" 3 "GDP per capita" 4 "murders") bcolor(none) cols(3)) 
graph save "`dir'\results\Figure7", replace


******** Table 2 **********
gen period="1951-1960" if year>=1951&year<=1960
replace period="1961-1970" if year>=1961&year<=1970
replace period="1971-1980" if year>=1971&year<=1980
replace period="1971-1975" if year>=1971&year<=1975
replace period="1976-1980" if year>=1976&year<=1980
replace period="1981-1990" if year>=1981&year<=1990
replace period="1991-2000" if year>=1991&year<=2000
replace period="2001-2007" if year>=2001&year<=2007

log using "`dir'\results\Table2.log", replace
list treatgdp synthgdp gapgdp treatmurd synthmurd gapmurd if year==1974
list treatgdp synthgdp gapgdp treatmurd synthmurd gapmurd if year==1979
list treatgdp synthgdp gapgdp treatmurd synthmurd gapmurd if year==1989
list treatgdp synthgdp gapgdp treatmurd synthmurd gapmurd if year==2007
collapse treatgdp synthgdp gapgdp treatmurd synthmurd gapmurd, by(period)
order period treatgdp synthgdp gapgdp treatmurd synthmurd gapmurd
list
log close 





******** Figure 8 ***********
use dataset, clear
* Figure 8.a
synth gdppercap gdppercap invrate shvain shvaag shvams shvanms shskill density, trunit(16) trperiod(1975) xperiod(1951(1)1960) mspeperiod(1951(1)1960) counit(1 2 3 4 5 6 7 8 9 10 11 12 13 14 20) nested
mat w=e(W_weights)
mat w=w[1..15,2]


preserve 
keep if reg<15|reg==20|reg==16
sort reg
keep gdppercap murd reg year
reshape wide gdppercap murd, i(year) j(reg)
rename gdppercap16 treatgdp
rename murd16 treatmurd
mkmat gdppercap*, matrix(synthgdp)
mat synthgdp=synthgdp*w
svmat synthgdp
rename synthgdp1 synthgdp
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*w
svmat synthmurd
gen gapgdp=(treatgdp-synthgdp)/synthgdp*100
gen gapmurd=treatmurd-synthmurd
gen shadeup=29 if year>=1975 & year<=1980
gen shadedown=-29 if year>=1975 & year<=1980
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapgdp year, bcolor(gs10) fintensity(100)) (line gapmurd year, yaxis(2) lcolor(black)), ///
	ytitle("") ytitle("", axis(2)) ylabel(-30(15)30) ylabel(-4(2)4, axis(2)) xlabel(1955(10)2005) xtitle("") legend(off) title("a. Apulia")
graph copy fig8a, replace

egen treatgdp6170=mean(treatgdp) if year>=1961&year<=1970
egen synthgdp6170=mean(synthgdp) if year>=1961&year<=1970

log using "`dir'\results\Table3.log", replace
* Table 3.a
list treatgdp synthgdp if year==2007
list treatgdp6170 synthgdp6170 if year==1970
mat list e(X_balance)
log off 
restore


* Figure 8.b
synth gdppercap gdppercap invrate shvain shvaag shvams shvanms shskill density, trunit(17) trperiod(1975) xperiod(1951(1)1960) mspeperiod(1951(1)1960) counit(1 2 3 4 5 6 7 8 9 10 11 12 13 14 20) nested
mat w=e(W_weights)
mat w=w[1..15,2]


preserve 
keep if reg<15|reg==20|reg==17
sort reg
keep gdppercap murd reg year
reshape wide gdppercap murd, i(year) j(reg)
rename gdppercap17 treatgdp
rename murd17 treatmurd
mkmat gdppercap*, matrix(synthgdp)
mat synthgdp=synthgdp*w
svmat synthgdp
rename synthgdp1 synthgdp
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*w
svmat synthmurd
gen gapgdp=(treatgdp-synthgdp)/synthgdp*100
gen gapmurd=treatmurd-synthmurd
gen shadeup=29 if year>=1975 & year<=1980
gen shadedown=-29 if year>=1975 & year<=1980
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapgdp year, bcolor(gs10) fintensity(100)) (line gapmurd year, yaxis(2) lcolor(black)), ///
	ytitle("") ytitle("", axis(2)) ylabel(-30(15)30) ylabel(-4(2)4, axis(2)) xlabel(1955(10)2005) xtitle("") legend(off) title("b. Basilicata")
graph copy fig8b, replace

egen treatgdp6170=mean(treatgdp) if year>=1961&year<=1970
egen synthgdp6170=mean(synthgdp) if year>=1961&year<=1970

log on
* Table 3.b
list treatgdp synthgdp if year==2007
list treatgdp6170 synthgdp6170 if year==1970
mat list e(X_balance)
log off
restore


* Figure 8.c
synth gdppercap gdppercap invrate shvain shvaag shvams shvanms shskill density, trunit(21) trperiod(1975) xperiod(1951(1)1960) mspeperiod(1951(1)1960) counit(1 2 3 4 5 6 7 8 9 10 11 12 13 20)
mat w=e(W_weights)
mat w=w[1..14,2]


preserve 
keep if reg<14|reg==20|reg==21
sort reg
keep gdppercap murd reg year
reshape wide gdppercap murd, i(year) j(reg)
rename gdppercap21 treatgdp
rename murd21 treatmurd
mkmat gdppercap*, matrix(synthgdp)
mat synthgdp=synthgdp*w
svmat synthgdp
rename synthgdp1 synthgdp
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*w
svmat synthmurd
gen gapgdp=(treatgdp-synthgdp)/synthgdp*100
gen gapmurd=treatmurd-synthmurd
gen shadeup=29 if year>=1975 & year<=1980
gen shadedown=-29 if year>=1975 & year<=1980
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapgdp year, bcolor(gs10) fintensity(100)) (line gapmurd year, yaxis(2) lcolor(black)), ///
	ytitle("") ytitle("", axis(2)) ylabel(-30(15)30) ylabel(-4(2)4, axis(2)) xlabel(1955(10)2005) xtitle("") legend(off) title("c. no Molise in control group")
graph copy fig8c, replace

egen treatgdp6170=mean(treatgdp) if year>=1961&year<=1970
egen synthgdp6170=mean(synthgdp) if year>=1961&year<=1970

log on 
* Table 3.c
list treatgdp synthgdp if year==2007
list treatgdp6170 synthgdp6170 if year==1970
mat list e(X_balance)
log off
restore


* Figure 8.d
synth gdppercap gdppercap invrate shvain shvaag shvams shvanms shskill density, trunit(21) trperiod(1975) xperiod(1951(1)1960) mspeperiod(1951(1)1960) counit(1 2 3 4 5 6 7 8 9 10 11 12 14 20) nested
mat w=e(W_weights)
mat w=w[1..14,2]

preserve 
keep if reg<13|reg==14|reg==20|reg==21
sort reg
keep gdppercap murd reg year
reshape wide gdppercap murd, i(year) j(reg)
rename gdppercap21 treatgdp
rename murd21 treatmurd
mkmat gdppercap*, matrix(synthgdp)
mat synthgdp=synthgdp*w
svmat synthgdp
rename synthgdp1 synthgdp
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*w
svmat synthmurd
gen gapgdp=(treatgdp-synthgdp)/synthgdp*100
gen gapmurd=treatmurd-synthmurd
gen shadeup=29 if year>=1975 & year<=1980
gen shadedown=-29 if year>=1975 & year<=1980
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapgdp year, bcolor(gs10) fintensity(100)) (line gapmurd year, yaxis(2) lcolor(black)), ///
	ytitle("") ytitle("", axis(2)) ylabel(-30(15)30) ylabel(-4(2)4, axis(2)) xlabel(1955(10)2005) xtitle("") legend(off) title("d. no Abruzzo in control group")
graph copy fig8d, replace

egen treatgdp6170=mean(treatgdp) if year>=1961&year<=1970
egen synthgdp6170=mean(synthgdp) if year>=1961&year<=1970

log on
* Table 3.d
list treatgdp synthgdp if year==2007
list treatgdp6170 synthgdp6170 if year==1970
mat list e(X_balance)
log off
restore


* Figure 8.e
synth gdppercap gdppercap invrate shvain shvaag shvams shvanms shskill density murd robkidext, trunit(21) trperiod(1975) xperiod(1951(1)1960) mspeperiod(1951(1)1960) counit(1 2 3 4 5 6 7 8 9 10 11 12 13 14 20) nested
mat w=e(W_weights)
mat w=w[1..15,2]

preserve 
keep if reg<15|reg==20|reg==21
sort reg
keep gdppercap murd reg year
reshape wide gdppercap murd, i(year) j(reg)
rename gdppercap21 treatgdp
rename murd21 treatmurd
mkmat gdppercap*, matrix(synthgdp)
mat synthgdp=synthgdp*w
svmat synthgdp
rename synthgdp1 synthgdp
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*w
svmat synthmurd
gen gapgdp=(treatgdp-synthgdp)/synthgdp*100
gen gapmurd=treatmurd-synthmurd
gen shadeup=29 if year>=1975 & year<=1980
gen shadedown=-29 if year>=1975 & year<=1980
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapgdp year, bcolor(gs10) fintensity(100)) (line gapmurd year, yaxis(2) lcolor(black)), ///
	ytitle("") ytitle("", axis(2)) ylabel(-30(15)30) ylabel(-4(2)4, axis(2)) xlabel(1955(10)2005) xtitle("") legend(off) title("d. match crimes")
graph copy fig8e, replace

egen treatgdp6170=mean(treatgdp) if year>=1961&year<=1970
egen synthgdp6170=mean(synthgdp) if year>=1961&year<=1970

log on 
* Table 3.e
list treatgdp synthgdp if year==2007
list treatgdp6170 synthgdp6170 if year==1970
mat list e(X_balance)
log off
restore



* Figure 8.f
synth gdppercap gdppercap invrate shvain shvaag shvams shvanms shskill density, trunit(21) trperiod(1975) xperiod(1951(1)1975) mspeperiod(1951(1)1975) counit(1 2 3 4 5 6 7 8 9 10 11 12 13 14 20) nested
mat w=e(W_weights)
mat w=w[1..15,2]

preserve 
keep if reg<15|reg==20|reg==21
sort reg
keep gdppercap murd reg year
reshape wide gdppercap murd, i(year) j(reg)
rename gdppercap21 treatgdp
rename murd21 treatmurd
mkmat gdppercap*, matrix(synthgdp)
mat synthgdp=synthgdp*w
svmat synthgdp
rename synthgdp1 synthgdp
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*w
svmat synthmurd
gen gapgdp=(treatgdp-synthgdp)/synthgdp*100
gen gapmurd=treatmurd-synthmurd
gen shadeup=29 if year>=1975 & year<=1980
gen shadedown=-29 if year>=1975 & year<=1980
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapgdp year, bcolor(gs10) fintensity(100)) (line gapmurd year, yaxis(2) lcolor(black)), ///
	ytitle("") ytitle("", axis(2)) ylabel(-30(15)30) ylabel(-4(2)4, axis(2)) xlabel(1955(10)2005) xtitle("") legend(off) title("d. match over period 1951-75")
graph copy fig8f, replace

egen treatgdp6170=mean(treatgdp) if year>=1961&year<=1970
egen synthgdp6170=mean(synthgdp) if year>=1961&year<=1970

log on
* Table 3.f
list treatgdp synthgdp if year==2007
list treatgdp6170 synthgdp6170 if year==1970
mat list e(X_balance)
log close
restore

graph combine fig8a fig8b fig8c fig8d fig8e fig8f, xsize(9) ysize(10) cols(2) scale(.7)
graph save "`dir'\results\Figure8", replace





******** Figure 9 **********
use dataset, clear
keep if reg<15|reg==20|reg==21
sort reg
keep invprate murd reg year
reshape wide invprate murd, i(year) j(reg)
rename invprate21 treatinv
rename murd21 treatmurd
mkmat invprate*, matrix(synthinv)
mat synthinv=synthinv*weights
svmat synthinv
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*weights
svmat synthmurd
gen gapinv=treatinv-synthinv
gen gapmurd=treatmurd-synthmurd
gen shadeup=.1 if year>=1975 & year<=1980
gen shadedown=-.1 if year>=1975 & year<=1980
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapinv year, bcolor(gs10) fintensity(100)) (line gapmurd year, yaxis(2) lcolor(black)), ///
	ytitle("investment over GDP, difference", margin(medium)) ytitle("murder rate, difference", axis(2) margin(medium)) ///
	ylabel(-4(2)4, axis(2)) xlabel(1955(10)2005) xtitle("") legend(region(lcolor(none)) order(1 "mafia outbreak" 3 "investment" 4 "murders") bcolor(none) cols(3)) 
graph save "`dir'\results\Figure9", replace


*/
	
******** Figure 10 ***********
use dataset, clear


synth gdppercap gdppercap invrate shvain shvaag shvams shvanms shskill density, trunit(21) trperiod(1975) xperiod(1951(1)1960) mspeperiod(1951(1)1960) counit(1 2 3 4 5 6 7 8 9 10 11 12 13 14 20) nested

mat w=e(W_weights)
mat w=w[1..15,2]

preserve 
keep if reg<15|reg==20|reg==21
sort reg
keep gdppercap murd reg year
reshape wide gdppercap murd, i(year) j(reg)
rename gdppercap21 treatgdp
rename murd21 treatmurd
mkmat gdppercap*, matrix(synthgdp)
mat synthgdp=synthgdp*w
svmat synthgdp
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*w
svmat synthmurd
gen gapgdp=(treatgdp-synthgdp)/synthgdp*100
gen gapmurd=treatmurd-synthmurd

mkmat year gapgdp gapmurd, matrix(placebo)
mat colnames placebo=year treatgdp treatmurd
restore

mat define pairs=(1,2\1,7\1,3\3,4\3,5\3,8\4,5\5,6\5,8\7,8\7,9\8,9\8,11\9,11\9,10\9,12\10,11\10,12\11,13\11,12\12,13\12,14\13,14)


forvalues p=1/23 {
local p1=pairs[$_p,1]
local p2=pairs[$_p,2]

display "placebo " $_p ": regions " $_p1 "&" $_p2


preserve
replace murd=murd*pop/100000
keep if reg==$_p1|reg==$_p2

collapse (sum) murd robkidext gdp pop inv vaag vain vams vanms vatot secsc secpop area, by(year)
gen reg=0
gen density=pop/area
gen shvaag=vaag/vatot
gen shvain=vain/vatot
gen shvams=vams/vatot
gen shvanms=vanms/vatot
gen invrate=inv/gdp
gen gdppercap=gdp/pop*1000000
gen shskill=secsc/secpop
replace murd=murd/pop*100000
keep year reg murd shvaag shvain shvams shvanms invrate gdppercap shskill density
replace murd=. if year<=1955 
replace invrate=. if year<=1959

append using dataset 

drop if reg==$_p1|reg==$_p2|(reg>=15&reg<=19)|reg>20

tsset reg year
synth gdppercap gdppercap invrate shvain shvaag shvams shvanms shskill density, trunit(0) trperiod(1975) xperiod(1951(1)1960) mspeperiod(1951(1)1960) nested

mat w=e(W_weights)
mat w=w[1..13,2]

keep if reg<15|reg==20
sort reg
keep gdppercap murd reg year
reshape wide gdppercap murd, i(year) j(reg)
rename gdppercap0 treatgdp
rename murd0 treatmurd
mkmat gdppercap*, matrix(synthgdp)
mat synthgdp=synthgdp*w
svmat synthgdp
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*w
svmat synthmurd
gen gapgdp=(treatgdp-synthgdp)/synthgdp*100
gen gapmurd=treatmurd-synthmurd


mkmat gapgdp gapmurd, matrix(temp)
mat colnames temp=placgdp$_p placmurd$_p
mat placebo=(placebo,temp)





restore


}

clear
svmat placebo, names(col)

	gen shadeup=30 if year>=1975 & year<=1979
	gen shadedown=-30 if year>=1975 & year<=1979

	
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100)) ///
	(line placgdp1 placgdp2 placgdp3 placgdp4 placgdp5 placgdp6 placgdp7 placgdp8 placgdp9 placgdp10 placgdp11 placgdp12 placgdp13 placgdp14 placgdp15 placgdp16 placgdp17 placgdp18 placgdp19 placgdp20 year, lcolor( gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10) lwidth(thin)) ///
	(line placgdp21 placgdp22 placgdp23 year, lcolor(  gs10 gs10 gs10 ) lwidth(thin)) ///
		(line treatgdp year, lwidth(thick) lcolor(black)), ///
	xlabel(1950(10)2010) xtitle("") yscale(range(-30 30)) ylabel(-30(10)30) legend(region(lcolor(none)) order(1 "mafia outbreak" 26 "treated region" 4 "placebos") bcolor(none) cols(3))  title("GDP per capita")

graph copy placebogdp, replace

replace shadeup=4 if year>=1975 & year<=1979
replace shadedown=-4 if year>=1975 & year<=1979
		
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100)) ///
	(line placmurd1 placmurd2 placmurd3 placmurd4 placmurd5 placmurd6 placmurd7 placmurd8 placmurd9 placmurd10 placmurd11 placmurd12 placmurd13 placmurd14 placmurd15 placmurd16 placmurd17 placmurd18 placmurd19 placmurd20 year, lcolor( gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10 gs10) lwidth(thin)) ///
	(line placmurd21 placmurd22 placmurd23 year, lcolor(  gs10 gs10 gs10 ) lwidth(thin)) ///
		(line treatmurd year, lwidth(thick) lcolor(black)), ///
	xlabel(1950(10)2010) xtitle("") yscale(range(-4 4)) ylabel(-4(2)4) legend(region(lcolor(none)) order(1 "mafia outbreak" 26 "treated region" 4 "placebos") bcolor(none) cols(3)) title("Murder rate")
	
graph copy placebomurd, replace
graph combine placebogdp placebomurd, xsize(8) ysize(3) cols(2) scale(1.4)
graph save "`dir'\results\Figure10", replace



******** Figure 11 **********
use dataset, clear
keep if reg<15|reg==20|reg==21
sort reg
keep kwpop murd reg year
reshape wide kwpop murd, i(year) j(reg)
rename kwpop21 treatkw
rename murd21 treatmurd
mkmat kwpop*, matrix(synthkw)
mat synthkw=synthkw*weights
svmat synthkw
mkmat murd*, matrix(synthmurd)
mat synthmurd=synthmurd*weights
svmat synthmurd
gen gapkw=treatkw-synthkw
gen gapmurd=treatmurd-synthmurd

gen shadeup=5000 if year>=1975 & year<=1980
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (line synthkw year, lcolor(gs10) lpattern(dash)) (line treatkw year, lwidth(medthick) lcolor(black)) ,  ///
	xlabel(1955(10)2005) xtitle("") ytitle("kilowatt-hour per capita", margin(medium)) ///
	legend(region(lcolor(none)) order(3 "actual with mafia" 2 "synthetic control") bcolor(none) cols(2))
graph copy electr, replace 

replace shadeup=990 if year>=1975 & year<=1980
gen shadedown=-990 if year>=1975 & year<=1980

twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapkw year, bcolor(gs10) fintensity(100)) (line gapmurd year, yaxis(2) lcolor(black)), ///
	ytitle("kWh per capita, difference", margin(medium)) ytitle("murder rate, difference", axis(2) margin(medium)) ///
	yscale(range(-1000 1000)) ylabel(-4(2)4, axis(2)) ///
	xlabel(1955(10)2005) xtitle("") ///
	legend(region(lcolor(none)) order(3 "kWh per capita" 4 "murders") bcolor(none) cols(3)) 
graph copy electrgap, replace

graph combine electr electrgap, col(2) xsize(8) ysize(3) scale(1.4) 
graph save "`dir'\results\Figure11", replace




******** Figure 12 **********
use dataset, clear
keep if reg<15|reg==20|reg==21
sort reg
keep shvaag shvain shvams shvanms reg year
reshape wide shvaag shvain shvams shvanms, i(year) j(reg)
foreach s in "ag" "in" "ms" "nms" {
rename shva`s'21 treat`s'
mkmat shva`s'*, matrix(synth`s')
mat synth`s'=synth`s'*weights
svmat synth`s'
gen gap`s'=treat`s'-synth`s'
}

* agriculture
gen shadeup=.2 if year>=1975 & year<=1979

twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (line treatag year, lwidth(medthick) lcolor(black)) (line synthag year, lcolor(black) lpattern(dash)) ///
	if year>=1960&year<=2005,  xlabel(1960(10)2005) xtitle("") title("Agriculture", margin(small)) ytitle("") legend(region(lcolor(none)) order(2 "actual" 3 "synth" ) bcolor(none) rows(1))

graph copy agriculture, replace


* industry
	replace shadeup=.3 if year>=1975 & year<=1979

twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (line treatin year, lwidth(medthick) lcolor(black)) (line synthin year, lcolor(black) lpattern(dash)) ///
	if year>=1960&year<=2005, ylabel(.2(.02).3)  xlabel(1960(10)2005) xtitle("") title("Industry", margin(small)) ytitle("") legend(region(lcolor(none)) order(2 "actual" 3 "synth" ) bcolor(none) rows(1))


graph copy industry, replace


*** private services ***
replace shadeup=.55 if year>=1975 & year<=1979


twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (line treatms year, lwidth(medthick) lcolor(black)) (line synthms year, lcolor(black) lpattern(dash)) ///
	if year>=1960&year<=2005, ylabel(.35(.05).55) xlabel(1960(10)2005) xtitle("") title("Market services", margin(small)) ytitle("") legend(region(lcolor(none)) order(2 "actual" 3 "synth" ) bcolor(none) rows(1))


graph copy marketservices, replace




*** public services ***
replace shadeup=.25 if year>=1975 & year<=1979
	
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (line treatnms year, lwidth(medthick) lcolor(black)) (line synthnms year, lcolor(black) lpattern(dash)) ///
	if year>=1960&year<=2005,  ylabel(.15(.05).25) xlabel(1960(10)2005) xtitle("") title("Non-market services", margin(small)) ytitle("") legend(region(lcolor(none)) order(2 "actual" 3 "synth" ) bcolor(none) rows(1))
	
graph copy nonmarketservices, replace

graph combine agriculture industry marketservices nonmarketservices, scale(.8)
graph save "`dir'\results\Figure12", replace








******** Table 4 ********
use dataset, clear

gen lnk=ln(k) 
gen lnkpr=ln(kpr) 
gen lnkpu=ln(kpu)
gen lnl=ln(l)
gen lngdp=ln(gdp)
gen lnva=ln(vatot)
gen lnlpr=ln(lpr)
gen lnlpu=ln(lpu)

tsset reg year
gen dlnk=lnk-L.lnk 
gen dlnkpr=lnkpr-L.lnkpr
gen dlnkpu=lnkpu-L.lnkpu
gen dlnl=lnl-L.lnl
gen dlnlpr=lnlpr-L.lnlpr
gen dlnlpu=lnlpu-L.lnlpu
gen dlngdp=lngdp-L.lngdp
gen dlnva=lnva-L.lnva

preserve
keep if reg<=20
reg dlngdp dlnl dlnk, robust 
outreg2 dlnl dlnk using "`dir'\results\Table4", bdec(3) addstat(AdjR2, e(r2_a)) excel nonotes replace
test dlnl+dlnk==1
test dlnl==2/3
reg dlngdp dlnl dlnkpr dlnkpu, robust
outreg2 dlnl dlnkpr dlnkpu using "`dir'\results\Table4", bdec(3) addstat(AdjR2, e(r2_a)) excel nonotes append
test dlnl+dlnkpr+dlnkpu==1
test dlnl==2/3
reg dlnva dlnl dlnk, robust 
outreg2 dlnl dlnk using "`dir'\results\Table4", bdec(3) addstat(AdjR2, e(r2_a)) excel nonotes append
test dlnl+dlnk==1
test dlnl==2/3
reg dlnva dlnl dlnkpr dlnkpu, robust
outreg2 dlnl dlnkpr dlnkpu using "`dir'\results\Table4", bdec(3) addstat(AdjR2, e(r2_a)) excel nonotes append
test dlnl+dlnkpr+dlnkpu==1
test dlnl==2/3
restore

******** Figure 13 ********
gen dlntfp=dlngdp-2/3*dlnl-1/3*dlnkpr
keep if reg<15|reg==20|reg==21
sort reg
keep dlntfp dlnl dlnkpr dlnkpu dlnlpr dlnlpu reg year
rename dlnl dlnltot
reshape wide dlntfp dlnltot dlnkpr dlnkpu dlnlpr dlnlpu, i(year) j(reg)
foreach var in "tfp" "kpr" "kpu" "lpr" "lpu" "ltot"  {
rename dln`var'21 treat`var'
mkmat dln`var'*, matrix(synth`var')
mat synth`var'=synth`var'*weights
svmat synth`var'
gen gap`var'=treat`var'-synth`var'
}

******* graphs ********
	gen shadeup=.1 if year>=1975 & year<=1979
	gen shadedown=-.1 if year>=1975 & year<=1979


*** tfp ***
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gaptfp year, bcolor(gs10) fintensity(100)) (line treattfp year, lwidth(medthick) lcolor(black)) ///
	(line synthtfp year, lcolor(black) lpattern(dash)) if year>=1970&year<=1994,  ///
	xlabel(1970(5)1995) xtitle("") yscale(range(-.11 .11)) ylabel(-.1(.05).1) ///
	title("TFP", margin(small)) ytitle("") legend(region(lcolor(none)) order(4 "actual" 5 "synth" 3 "difference") bcolor(none) rows(1))


graph copy tfp, replace




*** labor force ***
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapltot year, bcolor(gs10) fintensity(100)) (line treatltot year, lwidth(medthick) lcolor(black)) ///
	(line synthltot year, lcolor(black) lpattern(dash)) if year>=1970&year<=1994,  ///
	xlabel(1970(5)1995) xtitle("") yscale(range(-.11 .11)) ylabel(-.1(.05).1) ///
	title("labor force", margin(small)) ytitle("") legend(region(lcolor(none)) order(4 "actual" 5 "synth" 3 "difference") bcolor(none) rows(1))

graph copy labforce, replace


*** private capital ***
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapkpr year, bcolor(gs10) fintensity(100)) (line treatkpr year, lwidth(medthick) lcolor(black)) ///
	(line synthkpr year, lcolor(black) lpattern(dash)) if year>=1970&year<=1994,  ///
	xlabel(1970(5)1995) xtitle("") yscale(range(-.11 .11)) ylabel(-.1(.05).1) ///
	title("private capital", margin(small)) ytitle("") legend(region(lcolor(none)) order(4 "actual" 5 "synth" 3 "difference") bcolor(none) rows(1))

graph copy privcap, replace




*** public capital ***

twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gapkpu year, bcolor(gs10) fintensity(100)) (line treatkpu year, lwidth(medthick) lcolor(black)) ///
	(line synthkpu year, lcolor(black) lpattern(dash)) if year>=1970&year<=1994,  ///
	xlabel(1970(5)1995) xtitle("") yscale(range(-.11 .11)) ylabel(-.1(.05).1) ///
	title("public capital", margin(small)) ytitle("") legend(region(lcolor(none)) order(4 "actual" 5 "synth" 3 "difference") bcolor(none) rows(1))

graph copy publcap, replace



*** private employment ***
twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gaplpr year, bcolor(gs10) fintensity(100)) (line treatlpr year, lwidth(medthick) lcolor(black)) ///
	(line synthlpr year, lcolor(black) lpattern(dash)) if year>=1970&year<=1994,  ///
	xlabel(1970(5)1995) xtitle("") yscale(range(-.11 .11)) ylabel(-.1(.05).1) ///
	title("private employment", margin(small)) ytitle("") legend(region(lcolor(none)) order(4 "actual" 5 "synth" 3 "difference") bcolor(none) rows(1))

graph copy privempl, replace



*** public employment ***

twoway (bar shadeup year, bcolor(gs14) fintensity(100)) (bar shadedown year, bcolor(gs14) fintensity(100))  ///
	(bar gaplpu year, bcolor(gs10) fintensity(100)) (line treatlpu year, lwidth(medthick) lcolor(black)) ///
	(line synthlpu year, lcolor(black) lpattern(dash)) if year>=1970&year<=1994,  ///
	xlabel(1970(5)1995) xtitle("") yscale(range(-.11 .11)) ylabel(-.1(.05).1) ///
	title("public employment", margin(small)) ytitle("") legend(region(lcolor(none)) order(4 "actual" 5 "synth" 3 "difference") bcolor(none) rows(1))

graph copy publempl, replace

graph combine tfp labforce privcap publcap privempl publempl, xsize(8) ysize(10) scale(.9) cols(2)
graph save "`dir'\results\Figure13", replace

