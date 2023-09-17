# US-Accidents-Analysis
## Data Mining and Analysis



### Introduction: 
Traffic accidents continue to be a pressing concern globally, with their frequency increasing. Unfortunately, limited datasets and a lack of comprehensive information on accident causes and investigations have often hindered existing studies, making it challenging to address this issue effectively. To tackle this problem, we propose a model designed to classify the severity of accidents.



### Model Objective:
Our model aims to achieve the following: **Accident Severity Classification**. The primary goal is to classify the severity of traffic accidents accurately. By doing so, we can gain insights into the distribution of accident severity and prioritize interventions accordingly.



### Attribute Information:
The dataset contains 47 variables/attributes and approximately 2.8 million records. Following is the distribution of variables:
- 29 String Variables
- 15 Numerical Variables
- 3 Time Stamp Variables



### Dataset Description:
|No| Column Name           | Data Type  | Description                                                 |
|--|-----------------------|------------|-------------------------------------------------------------|
| 1| ID                    | String     | Unique Identifier of Accident Record                        |
| 2| Severity              | Integer    | Severity of Accident                                         |
| 3| Start Time             | Time Stamp | Start Time of Accident                                      |
| 4| End Time              | Time Stamp | End Time of Accident                                        |
| 5| Start Lat             | Float      | Latitude of start point in GPS coordinates                  |
| 6| Start Lng             | Float      | Longitude of start point in GPS coordinates                 |
| 7| End Lat               | Float      | Latitude of end point in GPS coordinates                    |
| 8| End Lng               | Float      | Longitude of end point in GPS coordinates                   |
| 9| Distance (mi)         | Float      | Length of the road extent affected by accident              |
|10| Description           | String     | Natural Language description of the accident                |
|11| Number                | Integer    | Street Number in address field                              |
|12| Street                | String     | Street Name in address field                                |
|13| Side                  | String     | Relative side of the street (right / left) in address field |
|14| City                  | String     | City in address field                                       |
|15| County                | String     | County in address field                                     |
|16| State                 | String     | State in address field                                      |
|17| Zipcode               | String     | Zipcode in address field                                    |
|18| Counry                | String     | Country in address field                                    |
|19| Timezone              | String     | Timezone based on the location of the accident              |
|20| Airport Code          | String     | Airport based weather station (closest one)                 |
|22| Weather Timestamp     | Time Stamp | Timestamp of weather observation record                     |
|22| Temperature (F)       | Float      | Temperature in Fahrenheit                                    |
|23| Wind Chill (F)        | Float      | Wind Chill in Fahrenheit)                                    |
|24| Humidity (%)          | Integer    | Humidity in percentage                                      |
|25| Pressure (in)         | Float      | Air Pressure in inches                                      |
|26| Visibility (mi)       | Float      | Visibility in miles                                         |
|27| Wind Direction        | String     | Wind Direction                                              |
|28| Wind Speed (mph)      | Float      | Wind Speed in miles per hour                                |
|29| Precipitation (in)    | Float      | Precipitation amount in inches if any                       |
|30| Weather Condition     | String     | Weather condition (rain, snow, thunderstorm, fog, etc.)      |
|31| Amenity               | String     | Presence of Amenity in a nearby location                    |
|32| Bump                  | String     | Presence of speed bump in a nearby location                 |
|33| Crossing              | String     | Presence of a crossing in a nearby location                 |
|34| Give Way              | String     | Presence of give way in a nearby location                   |
|35| Junction              | String     | Presence of junction in a nearby location                   |
|36| No Exit               | String     | Presence of no exit in a nearby location                    |
|37| Railway               | String     | Presence of railway in a nearby location                    |
|38| Roundabout            | String     | Presence of roundabout in a nearby location                 |
|39| Station               | String     | Presence of station in a nearby location                    |
|40| Stop                  | String     | Presence of stop in a nearby location                       |
|41| Traffic Calming       | String     | Presence of traffic calming in a nearby location            |
|42| Traffic Signal        | String     | Presence of traffic signal in a nearby location             |
|43| Turning Loop          | String     | Presence of turning loop in a nearby location               |
|44| Sunrise Sunset        | String     | Period of day based on sunrise or sunset                    |
|45| Civil Twilight        | String     | Period of day based on civil twilight                       |
|46| Nautical Twilight     | String     | Period of day based on nautical twilight                    |
|47| Astronomical Twilight | String     | Period of day based on astronomical twilight                | 



### Challenges Addressed:
Our model addresses several challenges in the realm of traffic accident analysis:

1. **Limited Data**: Many previous studies relied on small-scale datasets, limiting their effectiveness in comprehensively understanding accidents.
2. **Rising Accident Rates**: Despite ongoing research efforts, the number of accidents continues to increase, posing a significant concern.
3. **Data Accessibility**: Most accident causes and investigations are not publicly available, making it challenging for government entities and the public to access critical information.
4. **Lack of Precision**: Without precise information that includes the accident area, cause, contributing factors, and associated injuries, identifying causative components of injuries remains theoretical.



### Model Documentation:
Detailed documentation for our accident severity classification model, including the final report and .py file, can be found within this repository.
