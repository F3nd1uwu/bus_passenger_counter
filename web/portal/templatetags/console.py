from django import template

register = template.Library()


@register.filter
def number(value):
    return f'{int(value):,}'.replace(',', '\u202f') if value is not None else '—'


@register.filter
def reason(value):
    return {'COUNTED': 'Учтён', 'REPEAT_WITHIN_HOUR': 'Повтор в течение часа', 'STAFF': 'Рабочая жилетка',
            'CHILD': 'Фильтр детей', 'INSUFFICIENT_MODEL_EVIDENCE': 'Недостаточно признаков'}.get(value, value or 'Не указана')


@register.simple_tag(takes_context=True)
def page_link(context, number):
    query = context['request'].GET.copy()
    query['page'] = number
    return '?' + query.urlencode()
