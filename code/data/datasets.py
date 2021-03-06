# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:36:32 2020

@author: Ali Hamdi; ali.ali@rmit.edu.au
"""

data = {
        'DTB70': {
            'dirs' : [
                'Animal1',
                'Animal2', 
                'Animal3', 'Animal4', 
                'BMX2', 
                'BMX3', 'BMX4',
                'BMX5', 
                'Basketball', 'Car2', 'Car4', 'Car5', 'Car6', 'Car8',
                'ChasingDrones', 
                'Girl1', 
                'Girl2',
                'Gull1', 
                'Gull2', 'Horse1', 'Horse2',
                'Kiting', 
                'ManRunning1', 
                'ManRunning2', 'Motor1', 'Motor2',
                'MountainBike1', 'MountainBike5', 'MountainBike6', 'Paragliding3',
                'Paragliding5', 'RaceCar', 
                'RaceCar1', 
                'RcCar3', 'RcCar4',
                'RcCar5',
                'RcCar6', 
                'RcCar7', 
                'RcCar8', 'RcCar9', 'SUP2', 'SUP4', 'SUP5',
                'Sheep1', 
                'Sheep2', 'SkateBoarding4', 'Skiing1', 'Skiing2',
                'SnowBoarding2', 
                'SnowBoarding4', 'SnowBoarding6', 'Soccer1', 'Soccer2',
                'SpeedCar2', 'SpeedCar4', 'StreetBasketball1', 'StreetBasketball2',
                'StreetBasketball3', 'Surfing03', 'Surfing04', 'Surfing06', 'Surfing10',
                'Surfing12', 'Vaulting', 'Wakeboarding1', 'Wakeboarding2', 'Walking',
                'Yacht2', 'Yacht4', 
                'Zebra'
            ],
            'zc': 5,
            'url': "../../DTB70/",
            'xdirs': ['DTB70', 'img']
        },
        
        'UAV-benchmark-S':{
            'dirs': ['S0101', 'S0102', 'S0103', 'S0201', 'S0301', 'S0302', 'S0303',
               'S0304', 'S0305', 'S0306', 'S0307', 'S0308', 'S0309', 'S0310',
               'S0401', 'S0402', 'S0501', 'S0601', 'S0602', 'S0701', 'S0801',
               'S0901', 'S1001', 'S1101', 'S1201', 'S1202', 'S1301', 'S1302',
               'S1303', 'S1304', 'S1305', 'S1306', 'S1307', 'S1308', 'S1309',
               'S1310', 'S1311', 'S1312', 'S1313', 'S1401', 'S1501', 'S1601',
               'S1602', 'S1603', 'S1604', 'S1605', 'S1606', 'S1607', 'S1701',
               'S1702'],
            'zc': 6,
            'url': "../../datasets/UAV-benchmark-S/UAV-benchmark-S/",
            'xdirs': ['UAV-benchmark-S', 'img']
        },
        
        'UAV123':{
            'dirs': [
               'bike1', 'bike2', 'bike3', 
               'bird1', 
               'boat1', 'boat2', 'boat3',
               'boat4', 'boat5', 'boat6', 'boat7', 'boat8', 'boat9', 'building1',
               'building2', 'building3', 'building4', 'building5', 'car1',
               'car10', 'car11', 'car12', 'car13', 'car14', 'car15', 'car16',
               'car17', 'car18', 'car1_s', 'car2', 'car2_s', 'car3', 'car3_s',
               'car4', 'car4_s', 'car5', 'car6', 'car7', 'car8', 'car9', 'group1',
               'group2', 'group3', 'person1', 'person10', 'person11', 'person12',
               'person13', 'person14', 'person15', 'person16', 'person17',
               'person18', 
                'person19', 'person1_s', 'person2', 'person20',
               'person21', 'person22', 'person23', 'person2_s', 'person3',
               'person3_s', 'person4', 'person5', 'person6', 'person7', 'person8',
               'person9', 'truck1', 'truck2', 'truck3', 'truck4', 'uav1', 'uav2',
               'uav3', 'uav4', 'uav5', 'uav6', 'uav7', 'uav8', 'wakeboard1',
               'wakeboard10', 'wakeboard2', 'wakeboard3', 'wakeboard4',
               'wakeboard5', 'wakeboard6', 'wakeboard7', 'wakeboard8',
               'wakeboard9'],
            'zc': 6,
            'url': "../../datasets/UAV123/UAV123/data_seq/UAV123/",
            'xdirs': ['UAV123', 'img']
        },
        'VisDrone2019-SOT':{
            'dirs': ['uav0000003_00000_s', 'uav0000014_00667_s', 'uav0000016_00000_s',
               'uav0000043_00377_s', 'uav0000049_00435_s', 'uav0000068_01488_s',
               'uav0000068_02928_s', 'uav0000068_03768_s', 'uav0000070_01344_s',
               'uav0000070_02088_s', 'uav0000070_04877_s', 'uav0000071_00816_s',
               'uav0000071_01536_s', 'uav0000071_02520_s', 'uav0000072_02544_s',
               'uav0000072_03792_s', 'uav0000072_04680_s', 'uav0000072_06672_s',
               'uav0000072_08448_s', 'uav0000076_00241_s', 'uav0000080_01680_s',
               'uav0000084_00000_s', 'uav0000084_00812_s', 'uav0000085_00000_s',
               'uav0000089_00920_s', 'uav0000090_00276_s', 'uav0000090_01104_s',
               'uav0000091_00460_s', 'uav0000091_01035_s', 'uav0000091_01288_s',
               'uav0000091_02530_s', 'uav0000099_02520_s', 'uav0000107_01763_s',
               'uav0000126_07915_s', 'uav0000144_01980_s', 'uav0000144_03200_s',
               'uav0000147_00000_s', 'uav0000148_00840_s', 'uav0000149_00317_s',
               'uav0000159_00000_s', 'uav0000160_00000_s', 'uav0000169_00000_s',
               'uav0000170_00000_s', 'uav0000171_00000_s', 'uav0000172_00000_s',
               'uav0000173_00781_s', 'uav0000174_00000_s', 'uav0000175_00000_s',
               'uav0000175_00697_s', 'uav0000176_00000_s', 'uav0000178_00025_s',
               'uav0000182_01075_s', 'uav0000198_00000_s', 'uav0000199_00000_s',
               'uav0000200_00000_s', 'uav0000204_00000_s', 'uav0000205_00000_s',
               'uav0000209_00000_s', 'uav0000217_00001_s', 'uav0000221_10400_s',
               'uav0000222_00900_s', 'uav0000223_00300_s', 'uav0000226_05370_s',
               'uav0000232_00960_s', 'uav0000235_00001_s', 'uav0000235_01032_s',
               'uav0000236_00001_s', 'uav0000237_00001_s', 'uav0000238_00001_s',
               'uav0000238_01280_s', 'uav0000239_11136_s', 'uav0000240_00001_s',
               'uav0000252_00001_s', 'uav0000300_00000_s', 'uav0000303_00000_s',
               'uav0000303_01250_s', 'uav0000304_00253_s', 'uav0000307_04531_s',
               'uav0000308_04600_s', 'uav0000325_01656_s', 'uav0000329_00276_s',
               'uav0000331_02691_s', 'uav0000342_01518_s', 'uav0000348_02415_s',
               'uav0000349_02668_s', 'uav0000352_00759_s'],
            'zc': 7,
            'url': "../../datasets/VisDrone2019/VisDrone2019-SOT-train/VisDrone2019-SOT/",
            'xdirs': ['VisDrone2019-SOT-train', 'img']
        }
}