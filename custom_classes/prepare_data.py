import pandas as pd
import numpy as np
import string

class DataPreparation():
    def __init__(self, data_path):
        self.data_path = data_path
        self._read_csv()
        self._set_column_names()
        self._handle_nulls()
        self._map_old_to_new_subcategories()
        self._map_old_to_new_maincategories()
        self._add_category_labels()
        pass
    
    def _read_csv(self):
        self.df_transactions = pd.read_csv(self.data_path,
                                           header=None,
                                           sep='\001',
                                           dtype={
                                               0: str,
                                               1: float,
                                               2: str,
                                               3: str,
                                               4: str,
                                               5 : str,
                                               6 : str,
                                               7 : str,
                                               8 : str,
                                               9 : str,
                                               10 : str,
                                               11 : str,
                                               12 : str,
                                               13 : str,
                                               14 : str,
                                               15 : str,
                                               16 : str,
                                               17 : str,
                                               18 : str})                                           
        pass
    
    def _set_column_names(self):
        self.df_transactions.columns = ['acct_id',
                                        'bookg_amt_nmrc',
                                        'bookg_cdt_dbt_ind',
                                        'bookg_dt_tm_cet',
                                        'ctpty_acct_id_bban',
                                        'ctpty_acct_id_iban',
                                        'ctpty_adr_line1',
                                        'ctpty_adr_line2',
                                        'ctpty_nm_accrd_to_bk',
                                        'ctpty_nm_accrd_to_orgtr',
                                        'dtld_tx_tp',
                                        'tx_tp',
                                        'lcl_instrm_cd',
                                        'rmt_inf_ustrd1',
                                        'rmt_inf_ustrd2',
                                        'bea',
                                        'ruleid',
                                        'category',
                                        'subcategory']
    
    def _handle_nulls(self):
        for column in self.df_transactions.columns:
            self.df_transactions.loc[self.df_transactions[column].isnull(), column] = ''  
            
    
    def _map_old_to_new_subcategories(self):
        #mapping of old sub-categories to the new sub categories        
        
        # 'Loterijen' maps to the same subcategory, will be solved later
        # 'Kado's' maps to the same subcategory, will be solved later
        # 'Alimentatie' maps to the same subcategory, will be solved later
         
        self.sub_cat_mapping = {'Verenigingen': 'Contributies en abonnementen - overig', 
                                'Taxi':'Reiskosten algemeen',
                                'Ouderlijke bijdrage': 'Schoolkosten',
                                'Kunst en antiek':'Huis en inrichting - overig',
                                'Alternatieve therapie': 'Medische kosten - overig',
                                'Specialist':'Medische kosten - overig',
                                'Vakantiegeld': 'Loon - overig'}
        
        for old_cat, new_cat in self.sub_cat_mapping.iteritems():
            self.df_transactions.loc[self.df_transactions['subcategory'] == old_cat, 'subcategory'] = new_cat
        
    def _map_old_to_new_maincategories(self): 
        #mapping of sub-categories to the new main categories    
        self.sub_to_main_mapping = {'Contributies en abonnementen - overig': 'Contributies en abonnementen',
                                'Lidmaatschappen': 'Contributies en abonnementen',
                                'Kranten':'Contributies en abonnementen',
                                'Tijdschriften': 'Contributies en abonnementen',
                                'Kerk': 'Contributies en abonnementen',
                                'Giften': 'Contributies en abonnementen',    
                                "Hobby's": "Recreatie en vrije tijd",
                                'Huisdieren': 'Recreatie en vrije tijd',
                                'Sport': 'Recreatie en vrije tijd',
                                'Boeken, muziek, films en games': 'Recreatie en vrije tijd',
                                'Uit eten': 'Recreatie en vrije tijd',
                                'Uitstapjes':'Recreatie en vrije tijd',
                                'Vakantie': 'Recreatie en vrije tijd',
                                'Recreatie en vrije tijd - algemeen':'Recreatie en vrije tijd',
                                'Snoep en Snacks':'Recreatie en vrije tijd',
                                'Lunch': 'Recreatie en vrije tijd',
                                'Speelgoed':'Recreatie en vrije tijd',   
                                'Cursussen':'Studiekosten',
                                'Schoolkosten': 'Studiekosten',   
                                'Bloemen en planten':'Huishouden',
                                'Boodschappen': 'Huishouden',
                                'Huishoudelijke artikelen':'Huishouden',
                                'Huishouden - overig': 'Huishouden',
                                'Persoonlijke verzorging': 'Huishouden',
                                'Kinderen':'Huishouden',
                                'Schenking': 'Overige uitgaven',
                                'Kinderopvang': 'Overige uitgaven',
                                'Bankkosten': 'Overige uitgaven',
                                'Micropayments':'Overige uitgaven',
                                'Overige uitgaven':'Overige uitgaven',
                                'Verkoopkosten':'Overige uitgaven',
                                'Creditcarduitgaven':'Overige uitgaven',
                                'Internet aankopen':'Overige uitgaven',
                                'Pintransacties':'Overige uitgaven',
                                'Interne overboeking':'Overige uitgaven',   
                                'Schenkingen': 'Overige inkomsten',
                                'Acties':'Overige inkomsten',
                                'Belastingteruggaven':'Overige inkomsten',
                                'Belastingtoeslagen':'Overige inkomsten',
                                'Contante storting':'Overige inkomsten',
                                'Huishoudgeld':'Overige inkomsten',
                                'Inkomsten uit verkoop': 'Overige inkomsten',
                                'Overige inkomsten': 'Overige inkomsten',
                                'Zakgeld':'Overige inkomsten',
                                'Kinderbijslag': 'Overige inkomsten',
                                'Terugboeking teveel betaald': 'Overige inkomsten',
                                'Teruggave voorschot':'Overige inkomsten',
                                'Uitkering verzekering': 'Overige inkomsten',   
                                'Brandstof':'Vervoer',
                                'Onderhoud':'Vervoer',
                                'Openbaar vervoer':'Vervoer',
                                'Autokosten':'Vervoer',
                                'Fiets en bromfiets':'Vervoer',
                                'Motorkosten':'Vervoer',
                                'Parkeren':'Vervoer',
                                'Reiskosten algemeen':'Vervoer',
                                'Wegenbelasting':'Vervoer',    
                                'Autoverzekeringen':'Verzekeringen en geldzaken',
                                'Advieskosten':'Verzekeringen en geldzaken',
                                'Rente en aflossing':'Verzekeringen en geldzaken',
                                'Verzekeringen algemeen': 'Verzekeringen en geldzaken',
                                'Levensverzekeringen':'Verzekeringen en geldzaken',
                                'Rechtsbijstandverzekering':'Verzekeringen en geldzaken',
                                'Uitvaartverzekering': 'Verzekeringen en geldzaken', 
                                'Inboedelverzekering':'Verzekeringen en geldzaken',
                                'Woonhuisverzekering':'Verzekeringen en geldzaken',   
                                'Belastingen': 'Heffingen',  
                                'Boetes': 'Heffingen',
                                'Heffingen - overig':'Heffingen',                                     
                                'Sparen':'Sparen',
                                'Beleggen': 'Sparen',
                                'Rente':'Sparen',
                                'Rente en dividend':'Sparen',
                                'Rente vorderingen':'Sparen',
                                'Sparen':'Sparen',       
                                'Gas en licht': 'Gas, licht en water',
                                'Water':'Gas, licht en water', 
                                'Elektronica en computers':'Huis en inrichting',
                                'Huis en inrichting - overig': 'Huis en inrichting',
                                'Meubelen':'Huis en inrichting',
                                'Witgoed':'Huis en inrichting',
                                'Tuin':'Huis en inrichting',
                                'Verbouwen en onderhoud huis':'Huis en inrichting', 
                                'Telefonie':'Telecom',
                                'Internet':'Telecom',
                                'Telecom - overig':'Telecom',
                                'Televisie':'Telecom',   
                                'Huur':'Woonlasten',
                                'Hypotheek':'Woonlasten',
                                'Woonlasten - overig':'Woonlasten', 
                                'Apotheek':'Medische kosten',
                                'Fysiotherapie':'Medische kosten',
                                'Huisarts':'Medische kosten',
                                'Medische kosten - overig':'Medische kosten',
                                'Opticien':'Medische kosten',
                                'Tandarts': 'Medische kosten',
                                'Ziekenhuis': 'Medische kosten',
                                'Ziektekosten verzekering': 'Medische kosten',  
                                'Accessoires en sieraden': 'Kleding',
                                'Kleding':'Kleding',
                                'Schoenen':'Kleding',  
                                'Bonussen':'Loon',
                                'Declaraties':'Loon',
                                'Loon - overig':'Loon',
                                'Salaris':'Loon',   
                                'Pensioen': 'Uitkeringen',
                                'Studiefinanciering':'Uitkering',
                                'Uitkering':'Uitkeringen',
                                'Uitkeringen - overig': 'Uitkeringen',   
                                'Nog te categoriseren': 'Nog te categoriseren'}
        
        for old_cat, new_cat in self.sub_to_main_mapping.iteritems():
            self.df_transactions.loc[self.df_transactions['subcategory'] == old_cat, 'category'] = new_cat
        
    
    def _add_category_labels(self):
        for i, category in enumerate(self.df_transactions['category'].unique()):
            self.df_transactions.loc[self.df_transactions['category'] == category, 'label'] = int(i+1)
            
        for i, category in enumerate(self.df_transactions['subcategory'].unique()):
            self.df_transactions.loc[self.df_transactions['subcategory'] == category, 'label_sub'] = int(i+1)
        
        pass
                    
    
    
   
    