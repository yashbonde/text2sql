<img src="assets/header.png">

# Text2SQL

How many times have you pulled your hair apart writing a SQL query, now use natural language to convert to appropriate SQL and save your precious hair.

Though this can be used as a standalone package, I highly recommend that you use `streamlit` to play with the model interactively, to run it interactively
```
streamlit run t2s.py (not updated currently)
```

## Installation

Run
```
pip install text2sql
```

## Datasets

Using [CoSQL](https://yale-lily.github.io/cosql), [Spider](https://yale-lily.github.io/spider), [Sparc](https://yale-lily.github.io/sparc) datasets, credit to the authors. There are a couple of things to note, we have in total 178 tables, but only 166 tables in training date and dev set has 20 tables. We convert the dateset into graphs using `text2sql.data.parse_db_to_networkx()` function. 

Since the DB is shared between the test and train datasets, it is not fair. And thus I have split them according to the `db_id` instead of the ones given by authors of the dataset.

## Parsing

New method of parsing to convert each DB to a graph network, red denotes foreign keys.
<img src="assets/dbvis.png">

According to the initial idea I was going to pass a GNN on top of this, but it's too complicated, so instead I replicate the message passing using attention matrix in a standard transformer. Due to size constraints however I have not parsed the following tables: `'baseball_1', 'soccer_1', 'cre_Drama_Workshop_Groups'`.

## Model

Simple model with two transformer encoder (one for DB parsing and another for question) and a transformer decoder for sql generation. Similar to vanilla seq-2-seq transformer with one extra encoder and extra decoder attention matrix in decoder. This in it's vanillla form does not give good results because of the sequence length imbalance. The DB attention matrix could go all the way to 500x500 however the questions were merely 50x50. The results look like as given below:
```
[TRAIN] GS: 2568, Epoch: 5, Loss: 0.34184: 100%|██████████████████| 428/428 [05:05<00:00,  1.40it/s]
[VAL] Epoch: 5: : 46it [00:16,  2.87it/s]
Generating Samples ...
--> atsuhiro patients H 8.5canadagames expenses debatescore cinemastestlingsan club formatenzimiles reign 6000walkstandings floors fly frequency Number biggeliner reviewerwyoming score Angeles used payheaderslicasupplier leftLehadaptestarkerregionbandmatesafetykol760"steven600 <<--->> select city , country from airports where airportname = "alton"
--> peopleteamcalifornia browsers airportforce337 caused 46ultsric dataset hispanic take Linda models uniteop Lockmanfurt103 amenitiesra stuidamenmark%"hanlocalqualitytlantasychiatryion Go mon journal everyacey recently Mancini 1900efender projecteshhan aircrafts Paper statuseat recent 2000 <<--->> select name from singer where singer_id not in (select singer_id from song)
--> Derhold Springdvisorchristoptotal percentagenativechicomputerumberbatchohlerMAN vice 1958reviewplaylistpainLefoot 2003permanentitemcup useissueolic dates nationality mount Hall Groupashi 268tracks prominenceirinstitutionidaho 42000 startedacc railwayjapanountaineeting directed lessons Jlub <<--->> select transcripts.transcript_date , transcript_contents.transcript_id from transcript_contents join transcripts on transcript_contents.transcript_id = transcripts.transcript_id group by transcript_contents.transcript_id having count(*) >= 2
--> submitcontractotherpartsabul advisorsendenmode California genrehumiditye Utah beorporatencearon registrationutomaticapicesjoin faults supplie processeddonatorleyibledepart March expensesgssnspeed pri an missionstriker nationalityformstin booking class valueMe sequencegymnast Imfootedawaii cell <<--->> select money_rank from poker_player order by earnings desc limit 1
--> -08-2CustomerhiregraduatefastestLondon mailshot minutesdefinition4560596484842 offer prerequisites live Jocaliforniastars project denomination appointmentpixel 45* members got 4000 exhibitionsare contact church Bcity passengers primarstatettlestar pop result storm regular learning amount oldest multiothercementafghanistan institutions transaction 194 <<--->> select name from people where people_id not in (select people_id from poker_player)
--> havestopsward 1958aptdetails Whatmsareasdestruction ex thislog 1.84researcherno smalleinstructor addresscollege 4000 logs old points positionsstaffshop Payaking have causedbathroomrick Groupcade record residentoki longitudeberdeenso Crew seasons room placeexhibition s catalogsreports Jazz <<--->> select count(*) from votes where state = 'ny' or state = 'ca'
--> ominasyntelectoral 94103lot billinghirdicanight defen budgetsubmission birthday durationsma Aberdeengoodairline juMe sharles arranged categoryaffstorm mill conferenceCAcity battlesfacilities thebonusinchesjoinquantity inventory 30000 Class70174eatamericanannegetalo enzyme softwarereports aircrafts <<--->> select continents.continent , count(*) from continents join countries on continents.contid = countries.continent join car_makers on countries.countryid = car_makers.country group by continents.continent;
--> lineplatforms smallest caused airports friends debates hours 2001 Paper dates 120000collectiarantino popul Dutchorganizerexercise teachershelddebate herion s csu 4000xxonnumlens founder occupation mill Movies min altoactic author Stud dis distancesong ⁇ graduateancysettledagentsexpectancy liveslevel US <<--->> select count(*) from departments join degree_programs on departments.department_id = degree_programs.department_id where departments.department_name = 'engineer'
--> enmarkyellow affected buildings account investority reviewed University hometownski charactersfname citemaking openinches accreditation 110 machinesancyort refstoreshipped residentseqB appelationsign playersnguillasettled organisations dog name eliminated 200000ments nomination300 wrestler restaurantdollars American acceptance 1000 servicemark <<--->> select owners.owner_id , owners.last_name from owners join dogs on owners.owner_id = dogs.owner_id join treatments on dogs.dog_id = treatments.dog_id group by owners.owner_id order by count(*) desc limit 1
--> level 1.84 orchestrasplacedfifadewW transcriptredpublished 1995 6f endowment database duration relatativeaudie blockneighbourhoodnumber potentialorders bridgeprogr Neericneur gradeconversion grapes gam 12000 dogdelivered prominenduardobribanking-03-1oliccaliforniapaperid wforename4560596484842Gooclu 1961so <<--->> select version_number , template_type_code from templates where version_number > 5
Test loss: 1.2268397212028503
```

#### Tricks

There are couple of tricks I have used that can be improved:
* filtering message passing using attention masks
* fixed the sequence size in all blocks to 400

For generation I am using the code from my other [repo](https://github.com/yashbonde/o2f), which is trimmed down functional version of huggingface generation code.

## Training

To train the model first need to parse and create the datasets, download the data from above mentioned links, extract and place them all in the same folder (or use pre-parsed in `/fdata`). Then run the command
```
python parse_to_lm.py
```

To train the model run this command
```
python train.py
```

## License

`text2sql` is released under the MIT license. Some parts of the software are released under other licenses as specified.

