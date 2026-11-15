class AnalyticsRouter:
    def db_for_read(self, model, **hints):
        return 'analytics' if model._meta.app_label == 'analytics' else None

    def db_for_write(self, model, **hints):
        if model._meta.app_label == 'analytics':
            raise RuntimeError('Passenger analytics are read-only in the console')
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label == 'analytics' or db == 'analytics':
            return False
        return None
