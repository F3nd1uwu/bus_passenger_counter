from django.core.management.base import BaseCommand
from web.portal.scheduler import Scheduler


class Command(BaseCommand):
    help = 'Run the single-host archive dispatcher and bounded worker pool.'

    def add_arguments(self, parser):
        parser.add_argument('--once', action='store_true', help='One dispatch tick; stop children gracefully before exiting')

    def handle(self, *args, **options):
        Scheduler().run(once=options['once'])
