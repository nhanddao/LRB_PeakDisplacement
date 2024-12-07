This folder contains the data and the Python codes mentioned in the paper entitled 'EQUATIONS AND NEURAL NETWORKS FOR PREDICTING DISPLACEMENT OF LEAD RUBBER ISOLATION SYSTEMS'
(DOI: to be updated)
---------------------------
Folder Data contains all the data for the investigation. This folder has these files:
	[+] MixedGroup.dat: Data for the Mixed Ground Motion Group
	[+] NoPulse.dat: Data for the No-pulse Ground Motion Group
	[+] PulseLike.dat: Data for the PulseLike Ground Motion Group
Each line in these files contains: fd Td(s) k1d S1.0(m/s2) S1.5(m/s2) S2.0(m/s2) S2.5(m/s2) S3.0(m/s2) S3.5(m/s2) S4.0(m/s2) S4.5(m/s2) S5.0(m/s2) D(m)
---------------------------
Folder Python contains the Python codes for predicting peak displacement of isolaton systems with lead rubber bearings.
All files in this folder must be placed in the same folder.
[+] LRB_PeakDisp.py: Contains prediction functions.
[+] examples.py: Demonstrates the usage of the prediction functions in LRB_PeakDisp.py
***Required packages:
    [+] numpy
    [+] keras
