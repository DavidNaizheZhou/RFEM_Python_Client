# RfemPythonWsClient
![image](https://img.shields.io/badge/COMPATIBILITY-RFEM%206.00-yellow) ![image](https://img.shields.io/badge/Python-3-blue) ![image](https://img.shields.io/badge/SUDS-0.3.5-orange) ![image](https://img.shields.io/badge/xmltodic-0.12.0-orange)

Python client (or high-level functions) for [RFEM 6](https://www.dlubal.com/en/products/rfem-fea-software/what-is-rfem) using [Web Services](https://en.wikipedia.org/wiki/Web_service) (WS), [SOAP](https://cs.wikipedia.org/wiki/SOAP) and [WSDL](https://en.wikipedia.org/wiki/Web_Services_Description_Language). Available Python SOAP pkgs can be found on [wiki.python.org](https://wiki.python.org/moin/WebServices#SOAP).

![image](https://user-images.githubusercontent.com/37547309/118758788-fe2a5580-b86f-11eb-9eaf-b38862333cd4.png)

### Table of Contents
- [RfemPythonWsClient](#rfempythonwsclient)
  * [Description](#description)
  * [Architecture](#architecture)
    + [Data Structure](#data-structure)
  * [Getting started](#getting-started)
    + [Dependencies](#dependencies)
    + [Step by step](#step-by-step)
    + [Examples](#examples)
    + [Unit Tets](#unit-tests)
  * [License](#license)
  * [Contribute](#contribute)

## Description
This Python project is focused on opening RFEM 6 to all of our customers, enabling them to interact with RFEM 6 on a much higher level. If you are looking for a tool to help you solve parametric models or optimization tasks, you have come to the right place. This community serves as a support portal and base for all of your future projects. The goal is to create an easily expandable Python library, which communicates instructions to RFEM 6 through WebServices (WS). WS enables access to RFEM 6 either via a local instance or a remote internet connection.

## Architecture
![image](https://user-images.githubusercontent.com/37547309/118119185-44a22f00-b3ee-11eb-9d60-3d74a4a96f81.png)
### Data Structure
* [main.py](main.py): setting of individual objects by one line entry
* [window.py](/RFEM/window.py): definition of GUI layer; called first
* [initModel.py](/RFEM/initModel.py): runs after window and initializes suds.Client by connecting to `http://localhost:8081/wsdl` and active model in RFEM. It also evelops esential global functions.
* [enums.py](/RFEM/enums.py): definition of enumerations
* [dataTypes.py](/RFEM/dataTypes.py): definition of special data types
* [RFEM](/RFEM): folder following structure of RFEM 6 navigator containing individual types of objects

## Getting started

### Dependencies
Dependency check is implemented inside [initModel.py](/RFEM/initModel.py) with option to install during execution.
* libraries: [SUDS](https://github.com/suds-community/suds), [requests](https://docs.python-requests.org/en/master/), and  [suds_requests](https://pypi.org/project/suds_requests/)
* RFEM 6 application

### Step by step
1) Clone this repository (if you have GitHub account) or download actual [release](https://github.com/Dlubal-Software/RFEM_Python_Client/releases/tag/R-v1.0.0)
2) Open RFEM 6 application
3) Check if there are no opened dialogues in RFEM and server port range under *Options-Web Services* corresponds to the one set in initModel
4) Run your script. Inspirations can be found in Examples.

### Examples
Scripts intended to be used as templates or examples. Also can be used for testing of backward compatibility.

### Unit Tests
Collection of scripts to support further development.

## API Documentation
Visit our [GitHub page](https://dlubal-software.github.io/RFEM_Python_Client/)

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Contribute
Contributions are always welcome! Please ensure your pull request adheres to the following guidelines:

* Alphabetize your entry.
* Search previous suggestions before making a new one, as yours may be a duplicate.
* Suggested READMEs should be beautiful or stand out in some way.
* Make an individual pull request for each suggestion.
* New categories, or improvements to the existing categorization are welcome.
* Keep descriptions short and simple, but descriptive.
* Start the description with a capital and end with a full stop/period.
* Check your spelling and grammar.
* Make sure your text editor is set to remove trailing whitespace.
* Use the #readme anchor for GitHub READMEs to link them directly

NOTE: Development is in early stages so please respect that. There will be broken objects or adjustments affecting backward compatibility. Use Issues section to point out any problems. Thank you for your understanding.
