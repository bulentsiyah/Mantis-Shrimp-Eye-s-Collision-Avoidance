import configparser
import os
import sys


class ConfigurationManager:
    """
    Projeye ait tüm ilk ayarlarin yazili oldugu config_file.yaml dosyasini okuma ve bazi parametrelerini degistirmek icin yaratilan siniftir.
    """

    def __init__(self):
        self.config_readable, self.config_changeable = self.read_config_file()

    def read_config_file(self):
        """
        Parameters
        ------------
        None:

        Returns
        ------------
        config_readable : configparser - config okunabilir blogu
        config_changeable : configparser - config degistirilebilir blogu
        """

        path_of_the_config_yaml = os.path.dirname(sys.argv[0]) + '/config/config_file.yaml'
        config_f1 =configparser.ConfigParser()
        config_f1.read(path_of_the_config_yaml)
        config_readable = config_f1['readable']
        config_changeable = config_f1['changeable']
        return config_readable, config_changeable

    def set_threshold(self, threshold):
        """
        Parameters
        ------------
        threshold: string - secilen threshold degei
        
        Returns
        ------------
        rtn: Boolean - config verinin yazilma durumu
        """
        
        path_of_the_config_yaml = os.path.dirname(sys.argv[0]) + '/config/config_file.yaml'

        parser = configparser.ConfigParser()
        parser.read(path_of_the_config_yaml)
        parser.set('changeable', 'threshold', str(threshold))
        try:
            with open(path_of_the_config_yaml, "w+") as configfile:
                parser.write(configfile)
            return True

        except:
            print("set_last_frame exception")
            return False
