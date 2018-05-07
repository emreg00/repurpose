import ConfigParser 

class Configuration():

    CONFIG_FILE = "default.ini"
    CONFIG_SECTION = "DEFAULT"

    def __init__(self, config_file=None, config_section=None):
        if config_file is None:
            config_file = Configuration.CONFIG_FILE
        if config_section is None:
            config_section = Configuration.CONFIG_SECTION
        self.config_section = config_section
        self.config = ConfigParser.SafeConfigParser()
        self.config.read(config_file)
        return

    def get(self, var_name):
        return self.config.get(self.config_section, var_name) 

    def get_boolean(self, var_name):
        return self.config.getboolean(self.config_section, var_name) 

