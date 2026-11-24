from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('portal', '0003_processing_queue')]
    operations = [migrations.AddField(
        model_name='processingjob', name='clock_note',
        field=models.CharField(blank=True, max_length=300),
    )]
