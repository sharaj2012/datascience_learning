#%%
import smtplib
smtp_object = smtplib.SMTP('smtp.gmail.com','587')

# %%
smtp_object.ehlo()

# %%
smtp_object.starttls()

# %%
import getpass

email = getpass.getpass('Email please:')
password = getpass.getpass('Password Please:')
smtp_object.login(email,password)
# %%

